mod common;
mod jaro;
mod hamming;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;


#[polars_expr(output_type=Float64)]
fn jaro_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].utf8()?;
    let b = inputs[1].utf8()?;
    let out: Float64Chunked =
        arity::binary_elementwise_values(a, b, jaro::jaro_similarity);
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn jaro_winkler_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].utf8()?;
    let b = inputs[1].utf8()?;
    let out: Float64Chunked =
        arity::binary_elementwise_values(a, b, jaro::jaro_winkler_similarity);
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn jaro_winkler_similarity_longtol(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].utf8()?;
    let b = inputs[1].utf8()?;
    let out: Float64Chunked =
        arity::binary_elementwise_values(a, b, jaro::jaro_winkler_similarity_longtol);
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt32)]
fn hamming_distance(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].utf8()?;
    let b = inputs[1].utf8()?;
    let out: UInt32Chunked =
        arity::binary_elementwise_values(a, b, hamming::hamming_distance);
    Ok(out.into_series())
}