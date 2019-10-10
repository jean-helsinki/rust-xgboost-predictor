use crate::gbm::regtree::RegTree;
use crate::gbm::grad_booster::GradBooster;
use crate::model_reader::{ModelReader, ModelReadResult};
use crate::fvec::FVec;
use byteorder::LE;

struct ModelParam {
    /// number of trees
    num_trees: i32,
    /// number of root: default 0, means single tree
    num_roots: i32,
    /// number of features to be used by trees
    num_feature: i32,
    /// size of predicton buffer allocated used for buffering
    num_pbuffer: i64,
    /// how many output group a single instance can produce
    /// this affects the behavior of number of output we have:
    /// suppose we have n instance and k group, output will be k*n
    num_output_group: i32,
    /// size of leaf vector needed in tree
    size_leaf_vector: i32,
    /// reserved parameters
    reserved: [i32; 31],
}

impl ModelParam {
    fn new<T: ModelReader>(reader: &mut T) -> ModelReadResult<ModelParam> {
        let (num_trees, num_roots, num_feature) =
            (reader.read_i32_le()?, reader.read_i32_le()?, reader.read_i32_le()?);
        // read padding
        reader.read_i32_le()?;
        let num_pbuffer = reader.read_i64::<LE>()?;
        let (num_output_group, size_leaf_vector) = (reader.read_i32_le()?, reader.read_i32_le()?);
        let mut reserved = [0i32;31];
        reader.read_to_i32_buffer(&mut reserved)?;
        // read padding
        reader.read_i32_le()?;
        return Ok(ModelParam {
            num_trees,
            num_roots,
            num_feature,
            num_pbuffer,
            num_output_group,
            size_leaf_vector,
            reserved,
        });
    }

    pub fn pred_buffer_size(&self) -> i64 {
        return self.num_pbuffer * (self.num_output_group as i64)
            * ((self.size_leaf_vector + 1) as i64);
    }
}

/// Gradient boosted tree implementation
struct GBTree {
    mparam: ModelParam,
    trees: Vec<RegTree>,
    tree_info: Vec<i32>,
    group_trees: Vec<Vec<RegTree>>,
}

impl GBTree {
    fn parse_group_trees(num_output_group: usize, tree_info: &Vec<i32>, trees: &Vec<RegTree>) -> Vec<Vec<RegTree>> {
        return (0..num_output_group).map(|i|
            (0..tree_info.len())
                .filter(|j| tree_info[*j] == i as i32)
                .map(|j| trees[j].clone()).collect()
        ).collect()
    }

    fn new<T: ModelReader>(with_pbuffer: bool, reader: &mut T) -> ModelReadResult<Self> {
        let mparam = ModelParam::new(reader)?;
        let trees_result: ModelReadResult<Vec<RegTree>> = (0..mparam.num_trees).map(|_| RegTree::new(reader)).collect();
        let trees= trees_result?;

        let tree_info = reader.read_int_vec(mparam.num_trees as usize)?;

        if mparam.num_pbuffer != 0 && with_pbuffer {
            reader.skip(8 * mparam.pred_buffer_size() as usize)?;
        }

        let group_trees = GBTree::parse_group_trees(mparam.num_output_group as usize, &tree_info, &trees);

        return Ok(GBTree {
            mparam,
            trees,
            tree_info,
            group_trees,
        })
    }
}

impl<F: FVec> GradBooster<F> for GBTree {
    fn predict(feat: &F, ntree_limit: i32) -> Vec<f64> {
        unimplemented!()
    }

    fn predict_single(feat: &F, ntree_limit: i32) -> f64 {
        unimplemented!()
    }

    fn predict_leaf(feat: &F, ntree_limit: i32) -> Vec<i32> {
        unimplemented!()
    }
}
