mod common;

use assert_approx_eq::assert_approx_eq;
use std::collections::LinkedList;
use std::fs::File;
use xgboost_predictor::predictor::Predictor;

use crate::common::loaders::{load_data, load_expectation, open_resource_file};
use crate::common::tasks;
use crate::common::types::*;

fn verify(
    predictor: &TestPredictor,
    data: &LinkedList<DataItem>,
    expectation: LinkedList<Vec<f32>>,
    perform_predict_func: &Box<dyn Fn(&TestPredictor, &TestMap) -> Vec<f32>>,
) {
    assert_eq!(data.len(), expectation.len());
    for ((_, map), expected) in data.iter().zip(expectation.iter()) {
        let predicted = perform_predict_func(predictor, map);
        assert_eq!(predicted.len(), expected.len());
        for i in 0..predicted.len() {
            assert_approx_eq!(predicted[i], expected[i], 1e-5f32);
        }
    }
}

fn run(
    model_type: String,
    model_names: Vec<String>,
    data_file: String,
    tasks: Vec<tasks::PredictionTask>,
) {
    let data = load_data(&*format!("data/{}", data_file));
    for task in tasks.iter() {
        for model_name in model_names.iter() {
            let expectation = load_expectation(&*format!(
                "expectation/{}/{}.{}",
                model_type, model_name, task.expectation_suffix
            ));
            let mut model_file =
                open_resource_file(&*format!("model/{}/{}.model", model_type, model_name)).unwrap();
            let predictor: TestPredictor = Predictor::read_from::<File>(&mut model_file).unwrap();
            verify(&predictor, &data, expectation, &task.predict);
        }
    }
}

#[test]
fn test_gblinear() {
    run(
        String::from("gblinear"),
        vec![
            String::from("v40/binary-logistic"),
            String::from("v40/binary-logitraw"),
            String::from("v40/multi-softmax"),
            String::from("v40/multi-softprob"),
            String::from("v47/binary-logistic"),
            String::from("v47/binary-logitraw"),
            String::from("v47/multi-softmax"),
            String::from("v47/multi-softprob"),
        ],
        String::from("agaricus.txt.0.test"),
        vec![tasks::predict(), tasks::predict_margin()],
    )
}

#[test]
fn test_gbtree() {
    run(
        String::from("gbtree"),
        vec![
            String::from("v40/binary-logistic"),
            String::from("v40/binary-logitraw"),
            String::from("v40/multi-softmax"),
            String::from("v40/multi-softprob"),
            String::from("v47/binary-logistic"),
            String::from("v47/binary-logitraw"),
            String::from("v47/multi-softmax"),
            String::from("v47/multi-softprob"),
        ],
        String::from("agaricus.txt.0.test"),
        vec![
            tasks::predict(),
            tasks::predict_margin(),
            tasks::predict_with_ntree_limit(1),
            tasks::predict_with_excessive_ntree_limit(),
            tasks::predict_leaf(),
            tasks::predict_leaf_with_ntree(2),
        ],
    );
    run(
        String::from("gbtree"),
        vec![
            String::from("v40/binary-logistic"),
            String::from("v40/binary-logitraw"),
            String::from("v47/binary-logistic"),
            String::from("v47/binary-logitraw"),
        ],
        String::from("agaricus.txt.0.test"),
        vec![tasks::predict_single()],
    );
    run(
        String::from("gbtree"),
        vec![String::from("v47/rank-pairwise")],
        String::from("mq2008.test"),
        vec![
            tasks::predict(),
            tasks::predict_single(),
        ],
    );
    run(
        String::from("gbtree"),
        vec![String::from("v47/sms-spam")],
        String::from("sms-spam.test"),
        vec![
            tasks::predict(),
            tasks::predict_leaf(),
        ],
    );
}

#[test]
fn test_dart() {
    run(
        String::from("dart"),
        vec![String::from("rank-pairwise")],
        String::from("mq2008.test"),
        vec![
            tasks::predict(),
            tasks::predict_with_excessive_ntree_limit(),
        ],
    )
}
