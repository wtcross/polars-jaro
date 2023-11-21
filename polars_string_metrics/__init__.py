import polars as pl
from polars.type_aliases import IntoExpr
from polars.utils.udfs import _get_shared_lib_location

lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("similarity")
class Similarity:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def jaro(self, other: IntoExpr) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            args=[other],
            symbol="jaro_similarity",
            is_elementwise=True,
        )

    def jaro_winkler(self, other: IntoExpr) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            args=[other],
            symbol="jaro_winkler_similarity",
            is_elementwise=True,
        )

    def jaro_winkler_longtol(self, other: IntoExpr) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            args=[other],
            symbol="jaro_winkler_similarity_longtol",
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("distance")
class Distance:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def hamming(self, other: IntoExpr) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            args=[other],
            symbol="hamming_distance",
            is_elementwise=True,
        )