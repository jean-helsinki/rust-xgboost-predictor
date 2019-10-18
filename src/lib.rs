#![recursion_limit = "1024"]

#[macro_use]
extern crate error_chain;

mod errors {
    error_chain! {
        foreign_links {
            Io(::std::io::Error);
        }
    }
}

mod functions;
mod model_reader;
mod gbm;
pub mod fvec;
pub mod predictor;
