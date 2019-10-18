use crate::gbm::regtree::RegTree;
use crate::gbm::grad_booster::GradBooster;
use crate::model_reader::ModelReader;
use crate::fvec::FVec;
use crate::errors::*;


struct ModelParam {
    /// number of trees
    num_trees: i32,
    /// number of root: default 0, means single tree
    num_roots: i32,
    /// number of features to be used by trees
    num_feature: i32,
    /// size of predicton buffer allocated used for buffering
    num_pbuffer: usize,
    /// how many output group a single instance can produce
    /// this affects the behavior of number of output we have:
    /// suppose we have n instance and k group, output will be k*n
    num_output_group: usize,
    /// size of leaf vector needed in tree
    size_leaf_vector: usize,
    /// reserved parameters
    reserved: [i32; 31],
}

impl ModelParam {
    fn new<T: ModelReader>(reader: &mut T) -> Result<ModelParam> {
        let (num_trees, num_roots, num_feature) =
            (reader.read_i32_le()?, reader.read_i32_le()?, reader.read_i32_le()?);
        // read padding
        reader.read_i32_le()?;
        let num_pbuffer = reader.read_i64_le()? as usize;
        let (num_output_group, size_leaf_vector) =
            (reader.read_i32_le()? as usize, reader.read_i32_le()? as usize);
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

    pub fn pred_buffer_size(&self) -> usize {
        return self.num_pbuffer * (self.num_output_group)
            * (self.size_leaf_vector + 1);
    }
}

/// Gradient boosted tree implementation
pub struct GBTree {
    mparam: ModelParam,
    trees: Vec<RegTree>,
    tree_info: Vec<i32>,
    group_trees: Vec<Vec<RegTree>>,
    // use only in DART tree
    weight_drop: Option<Vec<f32>>,
}

impl GBTree {
    fn parse_group_trees(num_output_group: usize, tree_info: &Vec<i32>, trees: &Vec<RegTree>) -> Vec<Vec<RegTree>> {
        (0..num_output_group).map(|i|
            (0..tree_info.len())
                .filter(|j| tree_info[*j] == i as i32)
                .map(|j| trees[j].clone()).collect()
        ).collect()
    }

    pub fn new<T: ModelReader>(with_pbuffer: bool, reader: &mut T, is_dart: bool) -> Result<Self> {
        let mparam = ModelParam::new(reader)?;
        let trees_result: Result<Vec<RegTree>> = (0..mparam.num_trees).map(|_| RegTree::new(reader)).collect();
        let trees= trees_result?;

        let tree_info = reader.read_int_vec(mparam.num_trees as usize)?;

        if mparam.num_pbuffer != 0 && with_pbuffer {
            reader.skip(8 * mparam.pred_buffer_size())?;
        }

        let group_trees = GBTree::parse_group_trees(mparam.num_output_group as usize, &tree_info, &trees);

        let weight_drop = if is_dart {
            // if gbtree.mparam.num_trees != 0 {
            let size = reader.read_i64_le()? as usize;
            Some(reader.read_float_vec(size)?)
        } else { None };

        Ok(GBTree {
            mparam,
            trees,
            tree_info,
            group_trees,
            weight_drop,
        })
    }

    fn pred<F: FVec>(&self, feat: &F, bst_group: usize, root_index: usize, ntree_limit: usize) -> f64 {
        match self.weight_drop {
            None => {self.pred_as_gbtree(feat, bst_group, root_index, ntree_limit)},
            Some(weight_drop) => {self.pred_as_dart(feat, &weight_drop, bst_group, root_index, ntree_limit)},
        }
    }

    fn pred_as_dart<F: FVec>(&self, feat: &F, weight_drop: &Vec<f32>, bst_group: usize, root_index: usize, ntree_limit: usize) -> f64 {
        let trees = self.group_trees[bst_group].clone();
        assert!(ntree_limit <= trees.len());
        let treeleft = if ntree_limit == 0 { trees.len() } else { ntree_limit };
        (0..treeleft).map(|i| weight_drop[i] as f64 *trees[i].get_leaf_value(feat, root_index)).sum()
    }

    fn pred_as_gbtree<F: FVec>(&self, feat: &F, bst_group: usize, root_index: usize, ntree_limit: usize) -> f64 {
        let trees = self.group_trees[bst_group].clone();
        assert!(ntree_limit <= trees.len());
        let treeleft = if ntree_limit == 0 { trees.len() } else { ntree_limit };
        (0..treeleft).map(|i| trees[i].get_leaf_value(feat, root_index)).sum()
    }

    fn pred_path<F: FVec>(&self, feat: &F, root_index: usize, ntree_limit: usize) -> Vec<usize> {
        let treeleft = if ntree_limit == 0 {self.trees.len()} else {ntree_limit};
        (0..treeleft).map(|i| self.trees[i].get_leaf_index(feat, root_index)).collect()
    }
}

impl<F: FVec> GradBooster<F> for GBTree {
    fn predict(&self, feat: &F, ntree_limit: usize) -> Vec<f64> {
        (0..self.mparam.num_output_group)
            .map(|gid| self.pred(feat, gid as usize, 0, ntree_limit)).collect()
    }

    fn predict_single(&self, feat: &F, ntree_limit: usize) -> f64 {
        if self.mparam.num_output_group != 1 {
            panic!("Can't invoke predict_single() because this model outputs multiple values");
        }
        self.pred(feat, 0, 0, ntree_limit)
    }

    fn predict_leaf(&self, feat: &F, ntree_limit: usize) -> Vec<usize> {
        self.pred_path(feat, 0, ntree_limit)
    }
}
