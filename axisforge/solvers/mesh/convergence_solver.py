"""
axisforge/solvers/mesh/convergence_solver.py

Grid Convergence Index (Richardson extrapolation) math + per-interval
convergence bookkeeping. NÃO corre solves e NÃO sabe nada de
ShaftSystem/gears/bearings/SubmodelSolver -- só importa de results/
(mais stdlib/numpy). Isso é deliberado (confirmado com erg
2026-09-22): o loop de grades (chamar SubmodelSolver por grade_0,
grade_1, ...) e a resolução de "pontos" nomeados (ex: "centroid" de uma
carga distribuída, ponto de engrenamento de uma engrenagem) para um x
real ficam num dispatcher fora deste módulo -- este módulo só recebe,
já pronto, um SubmodelResult por nível de refinamento, mais os pontos
já resolvidos a float, e devolve se convergiu.

## REESCRITO nesta passagem (substituindo por completo o draft
## anterior, que estava desalinhado da arquitetura atual em 3 pontos
## de raiz -- ver conversa: SubmodelConvergencePostProcessing não
## existe; SubmodelSolver.solve() já devolve SubmodelResult
## pós-processado diretamente, não uma SubmodelSolution crua para
## reprocessar aqui; SubmodelResult.metric_values foi removido e
## substituído pelos arrays completos):
##
##   - metric_spec.py (MetricSpec, eval_points_for_interval,
##     default_displacement_metrics, *Strategy) -- REMOVIDO daqui.
##     Substituído por duas coisas INDEPENDENTES uma da outra:
##       1. "formas de avaliar" -- funções genéricas, registadas por
##          decorator (register_eval_form), que operam sobre um array
##          NUMPY simples (mais x_nodes/x quando precisam de um
##          ponto) -- nunca sabem de que variável (M, v, ...) o array
##          veio.
##       2. "variável" -- só o NOME do campo em SubmodelResult
##          (v_xz, M_xz, u, ...), usado com getattr() para ir buscar o
##          array antes de chamar a forma de avaliar.
##     O utilizador combina as duas em tuplos simples:
##       ("M_xz", "max_minus_mean")              -- sem ponto
##       ("v_xz", "at_point", "centroid")         -- com ponto
##     (motivo de serem independentes: o problema de convergência de M
##     na análise bearing-a-bearing NÃO era resolvido por avaliação
##     pontual -- o pico de M(x) desloca-se ligeiramente entre grades
##     -- e sim pelo desvio entre o máximo e a média de M sobre o
##     intervalo inteiro. Essa mesma forma de avaliar, max_minus_mean,
##     não tem nada de específico a M -- serve para qualquer variável.)
##   - SubmodelSolver/SubmodelSolution (submodel_solver/lagrange_multipliers.py)
##     -- REMOVIDO. O loop de grades sai para o dispatcher; este módulo
##     recebe SubmodelResult já resolvidos via add_level().
##   - SubmodelConvergencePostProcessing (submodel_solver/submodel_postprocessing.py)
##     -- REMOVIDO (nunca existiu como tal -- ver nota acima -- e mesmo
##     que existisse seria redundante: SubmodelResult que chega aqui já
##     está pós-processado).
##   - intervals_from_shaft_system() -- REMOVIDO daqui. Lia
##     shaft_system.gears/.bearings/.distributed_radial_loads e
##     axisforge.config.MIN_FACE_WIDTH_FOR_CONVERGENCE_MM -- nenhum dos
##     dois é results/. Passa a ser trabalho do dispatcher (é ele quem
##     já precisa de conhecer ShaftSystem para resolver os "pontos"
##     nomeados e para correr o SubmodelSolver).
##
## O que fica exatamente igual: RichardsonGCI e _DummyGCI -- já eram
## metric-agnostic (só recebem floats + listas de x_nodes), não tinham
## nenhum dos três imports banidos, não precisaram de mudar uma linha.

TODO(owner): as funções de "forma de avaliar" estão neste mesmo
ficheiro (max/mean/max_minus_mean/at_point) em vez de num
eval_forms.py à parte (ao estilo de contact_postprocessing.py separado
de contact_solver.py no iso_16281/) -- ainda não confirmaste se
preferes separado. Fácil de mover depois, mantido aqui por agora para
não multiplicar ficheiros antes de estabilizar a forma.

TODO(owner): peak_shifted (ConvergenceRecord, results/fem_results/convergence_results.py)
fica sempre vazio aqui -- nenhuma forma de avaliar devolve hoje "onde
está o extremo", só o valor. Se quiseres isto de volta, uma forma como
"max_minus_mean" teria de devolver (valor, x_do_extremo) em vez de só
valor, e add_level() teria de comparar esse x entre níveis -- não feito
nesta passagem.

References
----------
Roache, P.J. (1998). Verification and Validation in Computational Science and Engineering.
Richardson, L.F. (1911). Phil. Trans. R. Soc. London A, 210, 307-357.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np

from axisforge.results.fem_results.convergence_results import ConvergenceRecord, MeshRefinementResult

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.results.fem_results.submodel_results import SubmodelResult


# ===========================================================================
# Registo de "formas de avaliar" -- independentes de qualquer variável.
# Mesmo padrão de dispatch.py::register_contact_solver() (iso_16281/).
# ===========================================================================

EvalFormAggregate = Callable[[np.ndarray], float]
"""Assinatura para uma forma que agrega sobre o array inteiro -- não
precisa de saber onde é cada nó, só dos valores. Ex: max, mean,
max_minus_mean."""

EvalFormPoint = Callable[[np.ndarray, list, float], float]
"""Assinatura para uma forma que lê um único ponto -- recebe
(values, x_nodes, x). Ex: at_point."""

_EVAL_FORM_REGISTRY: dict[str, "EvalFormAggregate | EvalFormPoint"] = {}
_POINT_FORMS: set[str] = set()


def register_eval_form(name: str, *, needs_point: bool = False):
    """
    Decorator -- regista uma forma de avaliar sob um nome, independente
    de qualquer variável (a função nunca vê M/v/theta -- só o array de
    valores que o chamador já foi buscar com getattr()).

    needs_point : marca se a assinatura da função é
        (values, x_nodes, x) -- True -- ou só (values) -- False
        (default). add_level() usa isto para saber se tem de resolver
        um ponto nomeado para esta forma antes de a chamar.
    """
    def decorator(fn):
        if name in _EVAL_FORM_REGISTRY:
            raise ValueError(f"register_eval_form: '{name}' already registered")
        _EVAL_FORM_REGISTRY[name] = fn
        if needs_point:
            _POINT_FORMS.add(name)
        return fn
    return decorator


def resolve_eval_form(name: str):
    try:
        return _EVAL_FORM_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"resolve_eval_form: unknown eval form '{name}' -- "
            f"registered: {sorted(_EVAL_FORM_REGISTRY)}"
        ) from None


# --- formas concretas -- genéricas, nenhuma sabe de que variável veio o array ---

@register_eval_form("max")
def _eval_max(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


@register_eval_form("mean")
def _eval_mean(values: np.ndarray) -> float:
    return float(np.mean(values))


@register_eval_form("max_minus_mean")
def _eval_max_minus_mean(values: np.ndarray) -> float:
    """
    Desvio absoluto entre o máximo (em módulo) e a média, sobre o
    array inteiro -- a forma que resultou para M na análise
    bearing-a-bearing, quando avaliação pontual não resultava (o pico
    de M(x) desloca-se ligeiramente entre grades; comparar
    max-vs-mean é robusto a esse deslocamento, ao contrário de fixar
    um x)."""
    return float(np.max(np.abs(values)) - np.mean(values))


@register_eval_form("max_minus_mean_relative")
def _eval_max_minus_mean_relative(values: np.ndarray) -> float:
    """Versão normalizada de max_minus_mean -- ver essa docstring.
    Levanta se a média for exatamente 0.0 (caso degenerado, ex. um
    campo antissimétrico sobre o intervalo) em vez de propagar um
    ZeroDivisionError silencioso; _compute_gci() já sabe apanhar isto
    e cair para _DummyGCI, tal como para qualquer outro degenerado."""
    mean = float(np.mean(values))
    if mean == 0.0:
        raise ValueError("max_minus_mean_relative: mean is 0.0 -- cannot normalize")
    return float((np.max(np.abs(values)) - mean) / mean)


@register_eval_form("at_point", needs_point=True)
def _eval_at_point(values: np.ndarray, x_nodes: list, x: float) -> float:
    """Lê o valor no nó cuja posição bate com x -- x tem de já
    coincidir com um nó de x_nodes (o dispatcher é responsável por
    isso, o mesmo mecanismo de hard_points que SubmodelSolver.solve()
    já usa para garantir que um ponto pedido sobrevive ao dedupe da
    malha)."""
    for i, xi in enumerate(x_nodes):
        if abs(xi - x) < 1e-6:
            return float(values[i])
    raise ValueError(f"at_point: x={x} not found in x_nodes (tol=1e-6)")


# ===========================================================================
# GCI placeholder for the non-uniform-refinement-ratio case
# ===========================================================================

@dataclass
class _DummyGCI:
    GCI_f_m:   float = float("nan")
    GCI_m_c:   float = float("nan")
    converged: bool  = False


# ===========================================================================
# Richardson GCI -- inalterado desta passagem, já era metric-agnostic
# (só floats + listas de x_nodes, nunca um MetricSpec/solver/SubmodelResult).
# ===========================================================================

class RichardsonGCI:
    """
    Grid Convergence Index based on Richardson extrapolation.

    Requires a minimum of 3 consecutive refinement levels to compute
    the observed order of convergence p and the GCI per interval.
    Metric-agnostic -- takes f_coarse/f_medium/f_fine as plain floats.

    Parameters
    ----------
    f_coarse, f_medium, f_fine : valor da forma de avaliar em cada nível
    x_nodes_coarse/medium/fine : posições dos nós em cada nível
    gci_threshold   : fractional error threshold for convergence (default 0.01 = 1%)
    safety_factor   : Fs — 1.25 if p is verified in asymptotic range, 3.0 otherwise
    min_p, max_p    : clamp on observed order
    """

    def __init__(self,
                 f_coarse: float,
                 f_medium: float,
                 f_fine: float,
                 x_nodes_coarse: list,
                 x_nodes_medium: list,
                 x_nodes_fine: list,
                 gci_threshold: float = 0.01,
                 safety_factor: float = 1.25,
                 min_p: float | None = None,
                 max_p: float | None = None):

        self.x_nodes_coarse = x_nodes_coarse
        self.x_nodes_medium = x_nodes_medium
        self.x_nodes_fine   = x_nodes_fine
        self.gci_threshold  = gci_threshold
        self.safety_factor  = safety_factor
        self.f_coarse       = f_coarse
        self.f_medium       = f_medium
        self.f_fine         = f_fine

        n_c = len(x_nodes_coarse) - 1
        n_m = len(x_nodes_medium) - 1
        n_f = len(x_nodes_fine) - 1
        if n_c <= 0 or n_m <= 0 or n_f <= 0:
            raise ValueError(
                f"Cannot compute refinement ratio: level(s) with fewer "
                f"than 2 nodes (n_c={n_c}, n_m={n_m}, n_f={n_f})."
            )
        self.r_m_c = n_m / n_c
        self.r_f_m = n_f / n_m

        if self.r_m_c <= 1.0:
            raise ValueError(
                f"Refinement ratio r_m_c must be > 1. Got {self.r_m_c:.4f}. "
                f"Medium mesh must be finer than coarse mesh."
            )
        if self.r_f_m <= 1.0:
            raise ValueError(
                f"Refinement ratio r_f_m must be > 1. Got {self.r_f_m:.4f}. "
                f"Fine mesh must be finer than medium mesh."
            )
        if self.r_f_m != self.r_m_c:
            raise ValueError(
                f"Non-uniform refinement ratio: r_f_m={self.r_f_m:.4f} != "
                f"r_m_c={self.r_m_c:.4f}. "
                f"Grades must be generated by uniform bisection."
            )

        self.r = self.r_f_m

        scale = max(abs(self.f_coarse), abs(self.f_medium), abs(self.f_fine), 1.0)
        RELATIVE_FLAT_TOL = 1e-8
        ABS_FLAT_TOL = 1e-9
        d_m_c = abs(self.f_medium - self.f_coarse)
        d_f_m = abs(self.f_fine - self.f_medium)
        flat_tol = max(ABS_FLAT_TOL, RELATIVE_FLAT_TOL * scale)
        if d_m_c <= flat_tol and d_f_m <= flat_tol:
            self.p = float("nan")
            self.e_m_c = 0.0
            self.e_f_m = 0.0
            self.f_h0 = self.f_fine
            self.GCI_m_c = 0.0
            self.GCI_f_m = 0.0
            self.converged_m_c = True
            self.converged_f_m = True
            self.converged = True
            return

        if self.f_medium == self.f_fine:
            raise ValueError(
                "f_medium == f_fine -- cannot compute observed order p "
                "(division by zero in the convergence ratio) for this "
                "metric/interval."
            )
        ratio = (self.f_coarse - self.f_medium) / (self.f_medium - self.f_fine)
        if ratio <= 0.0:
            raise ValueError(
                f"Non-monotonic convergence (f_coarse-f_medium and "
                f"f_medium-f_fine have opposite sign, ratio={ratio:.6g} <= 0) "
                f"-- cannot compute a real observed order p for this "
                f"metric/interval."
            )
        self.p = float(np.log(ratio) / np.log(self.r))

        if min_p is not None:
            self.p = max(self.p, min_p)
        if max_p is not None:
            self.p = min(self.p, max_p)

        if abs(self.r**self.p - 1.0) < 1e-12:
            raise ValueError(
                f"Degenerate observed order p={self.p:.6g} (r**p - 1 ~= 0) -- "
                f"cannot extrapolate or compute GCI for this metric/interval."
            )

        if self.f_coarse == 0.0:
            raise ValueError(
                "f_coarse == 0.0 -- cannot compute relative error e_m_c "
                "(division by zero) for this metric/interval."
            )
        self.e_m_c = (self.f_medium - self.f_coarse) / self.f_coarse

        if self.f_medium == 0.0:
            raise ValueError(
                "f_medium == 0.0 -- cannot compute relative error e_f_m "
                "(division by zero) for this metric/interval."
            )
        self.e_f_m = (self.f_fine - self.f_medium) / self.f_medium

        self.f_h0 = self.f_coarse + (self.f_coarse - self.f_medium) / (self.r**self.p - 1)

        self.GCI_m_c = (self.safety_factor * abs(self.e_m_c) / (self.r_m_c**self.p - 1))
        self.GCI_f_m = (self.safety_factor * abs(self.e_f_m) / (self.r_f_m**self.p - 1))

        self.converged_m_c = self.GCI_m_c < self.gci_threshold
        self.converged_f_m = self.GCI_f_m < self.gci_threshold
        self.converged     = self.converged_f_m and self.converged_m_c


def _compute_gci(f_c, f_m, f_f, x_c, x_m, x_f, *,
                  gci_threshold: float, safety_factor: float,
                  min_p: float | None, max_p: float | None) -> "RichardsonGCI | _DummyGCI":
    """
    Metric-agnostic GCI computation with fallback. CHANGED nesta
    passagem: recebia um `spec: MetricSpec` e lia
    gci_threshold/safety_factor/min_p/max_p dele -- agora recebe-os
    diretamente como kwargs, porque MetricSpec deixou de existir.
    Isto significa que, tal como está, gci_threshold/safety_factor/
    min_p/max_p são os MESMOS para todos os pedidos (field, form[, point])
    de um MeshConvergenceStudy -- já não há override por-métrica
    individual como um MetricSpec permitia. Se precisares de thresholds
    diferentes por pedido, diz -- não implementado aqui porque o tuplo
    simples que escolheste não carrega essa informação.
    """
    try:
        return RichardsonGCI(
            f_coarse=f_c, f_medium=f_m, f_fine=f_f,
            x_nodes_coarse=x_c, x_nodes_medium=x_m, x_nodes_fine=x_f,
            gci_threshold=gci_threshold, safety_factor=safety_factor,
            min_p=min_p, max_p=max_p,
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        return _DummyGCI()


# ===========================================================================
# Request label -- deriva a chave de dicionário usada em
# ConvergenceRecord.point_metrics_history/gci_history a partir do tuplo
# (field, form[, point]).
# ===========================================================================

EvalRequest = tuple  # (field_name, form_name) ou (field_name, form_name, point_name)


def _request_label(request: EvalRequest) -> str:
    return ":".join(request)


# ===========================================================================
# Orchestrator -- SEM solves. Recebe um SubmodelResult por nível, já
# resolvido por um dispatcher; só faz a bookkeeping de convergência +
# GCI.
# ===========================================================================

class MeshConvergenceStudy:
    """
    Bookkeeping de convergência por intervalo, dado um SubmodelResult
    por nível de refinamento -- fornecido de fora (não corre nenhum
    solve). Ver módulo docstring para a divisão de responsabilidades
    com o dispatcher.

    Parameters
    ----------
    requests : list[tuple]
        Cada tuplo é (field_name, form_name) ou
        (field_name, form_name, point_name). field_name é lido de
        SubmodelResult via getattr(); form_name tem de estar registado
        via register_eval_form(); point_name só é necessário quando a
        forma tem needs_point=True, e o valor real (x) para esse nome
        vem do parâmetro `points` de add_level() -- resolvido pelo
        dispatcher, nunca aqui.
    gci_threshold, safety_factor, min_p, max_p :
        Aplicados a TODOS os pedidos -- ver nota em _compute_gci()
        sobre a perda do override por-métrica que MetricSpec permitia.
    min_levels_for_gci : Richardson requires 3 evaluations minimum.
    """

    _MIN_LEVELS_FOR_GCI = 3

    def __init__(
        self,
        requests: list[EvalRequest],
        gci_threshold: float = 0.01,
        safety_factor: float = 1.25,
        min_p: float | None = None,
        max_p: float | None = None,
    ):
        self._requests = list(requests)
        for request in self._requests:
            form_name = request[1]
            resolve_eval_form(form_name)  # valida cedo -- falha já na construção, não a meio de um estudo
            needs_point = form_name in _POINT_FORMS
            if needs_point and len(request) != 3:
                raise ValueError(
                    f"MeshConvergenceStudy: request {request} uses form "
                    f"'{form_name}', which needs a point -- expected "
                    f"(field_name, form_name, point_name)."
                )
            if not needs_point and len(request) != 2:
                raise ValueError(
                    f"MeshConvergenceStudy: request {request} uses form "
                    f"'{form_name}', which takes no point -- expected "
                    f"(field_name, form_name)."
                )

        self._gci_threshold = gci_threshold
        self._safety_factor = safety_factor
        self._min_p = min_p
        self._max_p = max_p

        # bookkeeping privado -- x_nodes por nível, por record. Não
        # guardado em ConvergenceRecord (não é um campo desse dataclass
        # e não fui autorizado a alterar convergence_results.py) --
        # vive só durante a vida desta instância de MeshConvergenceStudy,
        # chaveado por id(rec). Pressupõe que cada ConvergenceRecord é
        # processado até ao fim (converged ou max_levels esgotados)
        # antes de o objeto ser descartado -- não pensado para
        # sobreviver a um record reconstruído com o mesmo id() depois
        # de o original ser garbage-collected.
        self._x_nodes_history: dict[int, list[list[float]]] = {}

    def new_record(self, label: str, x_lo: float, x_hi: float) -> ConvergenceRecord:
        return ConvergenceRecord(label=label, x_lo=x_lo, x_hi=x_hi)

    def add_level(
        self,
        rec: ConvergenceRecord,
        grade: str,
        result: "SubmodelResult",
        points: dict[str, float] | None = None,
    ) -> bool:
        """
        Regista mais um nível de refinamento em `rec`. Devolve True se
        esta atualização atingiu convergência (rec.converged e
        rec.x_final ficam definidos) -- o dispatcher usa o retorno para
        decidir se para o loop de grades.

        points : {point_name: x} para os pedidos que precisam de ponto
            -- já resolvido (ex: "centroid" -> 47.3) por quem chama.
            Ausente/None se nenhum pedido desta study precisar de ponto.
        """
        metrics: dict[str, float] = {}
        for request in self._requests:
            field_name, form_name = request[0], request[1]
            values = getattr(result, field_name)
            form = resolve_eval_form(form_name)

            if form_name in _POINT_FORMS:
                point_name = request[2]
                if points is None or point_name not in points:
                    raise ValueError(
                        f"add_level: request {request} needs point "
                        f"'{point_name}', not provided in `points`."
                    )
                value = form(values, result.x_nodes, points[point_name])
            else:
                value = form(values)

            metrics[_request_label(request)] = value

        rec.levels.append(grade)
        rec.point_metrics_history.append(metrics)

        x_history = self._x_nodes_history.setdefault(id(rec), [])
        x_history.append(list(result.x_nodes))

        if len(rec.levels) >= self._MIN_LEVELS_FOR_GCI:
            gci_this_transition: dict[str, "RichardsonGCI | _DummyGCI"] = {}
            for label in metrics:
                f_c = rec.point_metrics_history[-3][label]
                f_m = rec.point_metrics_history[-2][label]
                f_f = rec.point_metrics_history[-1][label]
                x_c, x_m, x_f = x_history[-3], x_history[-2], x_history[-1]
                gci_this_transition[label] = _compute_gci(
                    f_c, f_m, f_f, x_c, x_m, x_f,
                    gci_threshold=self._gci_threshold,
                    safety_factor=self._safety_factor,
                    min_p=self._min_p, max_p=self._max_p,
                )
            rec.gci_history.append(gci_this_transition)

            if all(g.converged for g in gci_this_transition.values()):
                rec.converged = True
                rec.x_final = list(result.x_nodes)
                return True

        return False

    def finalize_unconverged(self, rec: ConvergenceRecord, result: "SubmodelResult") -> None:
        """
        Chamar quando max_levels foi esgotado sem convergir --
        replica o fallback do draft antigo ("if history: rec.x_final =
        list(history[-1][2].x_nodes)"), incluindo a mesma ressalva já
        documentada em MeshRefinementResult.all_extra_nodes: isto
        publica x_final mesmo sem rec.converged==True. Chamador
        (dispatcher) decide se quer mesmo isso -- não mudado aqui, só
        exposto explicitamente em vez de ficar escondido dentro de um
        loop.
        """
        rec.x_final = list(result.x_nodes)


def build_mesh_refinement_result(shaft_name: str, per_load: dict[str, ConvergenceRecord]) -> MeshRefinementResult:
    """Pequeno helper -- monta o MeshRefinementResult final a partir
    dos ConvergenceRecord já preenchidos pelo dispatcher, um por
    intervalo. Não faz mais nada além de instanciar a dataclass."""
    return MeshRefinementResult(shaft_name=shaft_name, per_load=per_load)