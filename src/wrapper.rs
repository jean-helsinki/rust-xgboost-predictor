use pyo3::prelude::*;

use crate::fvec::FVecArray;
use crate::predictor::Predictor;

use std::time::{Duration, Instant};

#[pyclass]
pub struct PredictorWrapper {
    pub predictor: Predictor<FVecArray<f32>>,
}

#[pymethods]
impl PredictorWrapper {
    #[args(ntree_limit = "0", margin = "false")]
    pub fn predict(&self, data: Vec<f32>, ntree_limit: usize, margin: bool) -> PyResult<Vec<f32>> {
        Ok(self
            .predictor
            .predict(&FVecArray::new(data), margin, ntree_limit))
    }

    #[args(ntree_limit = "0", margin = "false")]
    pub fn predict_batch(
        &self,
        data: Vec<Vec<f32>>,
        ntree_limit: usize,
        margin: bool,
    ) -> PyResult<Vec<Vec<f32>>> {
        let mut tranformed = vec![];
        for row in data.into_iter() {
            tranformed.push(FVecArray::new(row))
        }
        Ok(self
            .predictor
            .predict_many(&tranformed, true, ntree_limit))
    }

    #[args(ntree_limit = "0", margin = "false")]
    pub fn predict_single(
        &self,
        data: Vec<f32>,
        ntree_limit: usize,
        margin: bool,
    ) -> PyResult<f32> {
        Ok(self
            .predictor
            .predict_single(&FVecArray::new(data), margin, ntree_limit))
    }

    #[args(ntree_limit = "0")]
    pub fn predict_leaf(&self, data: Vec<f32>, ntree_limit: usize) -> PyResult<Vec<usize>> {
        Ok(self
            .predictor
            .predict_leaf(&FVecArray::new(data), ntree_limit))
    }
}
