import polars as pl
from polars.type_aliases import IntoExpr
from polars.utils.udfs import _get_shared_lib_location

lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("jaro")
class Jaro:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def jaro_similarity(self, other: IntoExpr) -> pl.Expr:
        return self._expr._register_plugin(
            lib=lib,
            args=[other],
            symbol="jaro_similarity",
            is_elementwise=True,
        )

    def jaro_winkler_similarity(self, other: IntoExpr) -> pl.Expr:
        return self._expr._register_plugin(
            lib=lib,
            args=[other],
            symbol="jaro_winkler_similarity",
            is_elementwise=True,
        )

    def jaro_winkler_similarity_longtol(self, other: IntoExpr) -> pl.Expr:
        return self._expr._register_plugin(
            lib=lib,
            args=[other],
            symbol="jaro_winkler_similarity_longtol",
            is_elementwise=True,
        )