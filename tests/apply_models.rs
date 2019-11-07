mod common;

use std::path::PathBuf;
use std::fs::File;
use std::io::BufReader;
use xgboost_predictor::predictor::Predictor;
use std::collections::LinkedList;

use crate::common::loaders::{open_resource_file, load_data, load_expectation};
use crate::common::types::*;

fn predict_and_log_loss(predictor: &TestPredictor, data: LinkedList<DataItem>) -> f32 {
    let mut sum = 0f32;
    for (actual, map) in data.iter() {
        let predicted = predictor.predict(map, false, 0);
        let sub_zero = 1e-15f32;
        let pred_value = predicted[0].max(sub_zero).min(1f32 - sub_zero);

    };
    0f32
}

fn predict_leaf_index(predictor: &TestPredictor, data: LinkedList<DataItem>) {

}

#[test]
fn test_predict() {
    let data = load_data("data/agaricus.txt.0.test");
    let expectation = load_expectation("expectation/gblinear/v40/binary-logistic.predict");
    let mut model_file = open_resource_file("model/gbtree/v47/binary-logistic.model").unwrap();
    let predictor: TestPredictor = Predictor::read_from::<File>(&mut model_file).unwrap();
    predict_and_log_loss(&predictor, data);
}