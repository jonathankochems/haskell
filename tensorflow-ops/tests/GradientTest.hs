-- Copyright 2016 TensorFlow authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE ScopedTypeVariables #-}

import Control.Monad(forM, forM_, replicateM)
import Data.Int (Int32,Int64)
import Data.List (sort)
import qualified Data.List as List
import Data.ProtoLens.TextFormat (showMessage)
import Google.Test (googleTest)
import Lens.Family2 ((^..))
import Test.Framework (Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?))
import qualified Data.Vector as V
import System.Random (randomIO)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF (max)
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Ops as TF

import Proto.Tensorflow.Core.Framework.Graph (node)
import Proto.Tensorflow.Core.Framework.NodeDef (op)

testGradientSimple :: Test
testGradientSimple = testCase "testGradientSimple" $ do
    let grads = do
                x <- TF.render $ TF.scalar (3 :: Float)
                b <- TF.render $ TF.scalar (4 :: Float)
                let y = x `TF.mul` x `TF.add` b
                TF.gradients y [x, b]
    -- Assert that the gradients are right.
    [dx, db] <- TF.runSession $ grads >>= TF.run
    6 @=? TF.unScalar dx
    1 @=? TF.unScalar db
    -- Assert that the graph has the expected ops.
    let graphDef = TF.asGraphDef grads
    putStrLn $ showMessage graphDef
    let ops = graphDef ^.. node . traverse . op
        expected = [ "Const"
                   , "Mul"
                   , "Const"
                   , "Add"
                     -- Default output gradient of y.
                   , "Shape"
                   , "Const"
                   , "Fill"
                     -- Add gradient.
                   , "Shape"
                   , "Shape"
                   , "BroadcastGradientArgs"
                   , "Sum"
                   , "Sum"
                   , "Reshape"
                   , "Reshape"
                     -- Mul gradient.
                   , "Shape"
                   -- This Op gets dedup'd because the inputs are the same.
                   -- TODO(fmayle): The same would happen to the Mul and Sum ops
                   -- below if the gradient function didn't multiply one as
                   -- 'dz * y' and the other as 'x * dz'. We could change the
                   -- order, but I'm going to keep it the same as the python
                   -- version for now.
                   --
                   -- , "Shape"
                   , "BroadcastGradientArgs"
                   , "Mul"
                   , "Mul"
                   , "Sum"
                   , "Sum"
                   , "Reshape"
                   , "Reshape"
                     -- AddN to combine x's output gradients.
                   , "AddN"
                   ]
    sort expected @=? sort ops

testGradientDisconnected :: Test
testGradientDisconnected = testCase "testGradientDisconnected" $ do
    let grads = do
            x <- TF.render $ TF.scalar (3 :: Float)
            b <- TF.render $ TF.scalar (4 :: Float)
            TF.gradients x [x, b]
    -- Assert that the gradients are right.
    [dx, db] <- TF.runSession $ grads >>= TF.run
    1 @=? TF.unScalar dx
    0 @=? TF.unScalar db
    -- Assert that the graph has the expected ops.
    let graphDef = TF.asGraphDef grads
    putStrLn $ showMessage graphDef
    let ops = graphDef ^.. node . traverse . op
        expected = [ "Const"
                   , "Const"
                     -- Default output gradient of x.
                   , "Shape"
                   , "Const"
                   , "Fill"
                     -- Default output gradient of b.
                   , "ZerosLike"
                   ]
    sort expected @=? sort ops


-- Test that identical "stateful" ops work with createGraph.
testCreateGraphStateful :: Test
testCreateGraphStateful = testCase "testCreateGraphStateful" $ do
    [dx, dy] <- TF.runSession $ do
        let shape = TF.constant (TF.Shape [1]) [1]
        x :: TF.Tensor TF.Value Float <- TF.truncatedNormal shape
        y :: TF.Tensor TF.Value Float <- TF.truncatedNormal shape
        TF.gradients (TF.expr x + TF.expr y * 3) [x, y] >>= TF.run
    -- If this test fails, it will likely be caused by an exception within
    -- `TF.gradients`. These asserts are extra.
    1 @=? TF.unScalar dx
    3 @=? TF.unScalar dy


-- Test that name scopes work with createGraph.
testCreateGraphNameScopes :: Test
testCreateGraphNameScopes = testCase "testCreateGraphNameScopes" $ do
    [dx] <- TF.runSession $ do
        let shape = TF.constant (TF.Shape [1]) [1]
        x :: TF.Tensor TF.Value Float <-
            TF.withNameScope "foo" (TF.truncatedNormal shape)
        TF.gradients x [x] >>= TF.run
    -- If this test fails, it will likely be caused by an exception within
    -- `TF.gradients`. This assert is extra.
    1 @=? TF.unScalar dx


-- Test that createGraph can handle graphs with diamond shapes.
testDiamond :: Test
testDiamond = testCase "testDiamond" $ do
    [dx] <- TF.runSession $ do
        x <- TF.render $ TF.vector [1]
        let y = x `TF.mul` x
            z = y*y
        TF.gradients z [x] >>= TF.run
    (4 :: Float) @=? TF.unScalar dx


testMaxGradient :: Test
testMaxGradient = testCase "testMaxGradient" $ do
    [dx] <- TF.runSession $ do
        x <- TF.render $ TF.vector [1, 2, 3, 0, 1 :: Float]
        let y = TF.max x (0 :: TF.Tensor TF.Build Int32)
        TF.gradients y [x] >>= TF.run
    V.fromList [0, 0, 1, 0, 0 :: Float] @=? dx

testConcatGradient :: Test
testConcatGradient = testCase "testConcatGradient" $ do
    [dv,dv'] <- TF.runSession $ do
        v  <- TF.render $ TF.vector [1 :: Float]
        v' <- TF.render $ TF.vector [2 :: Float]
        let y = TF.concat (TF.scalar 0) [ v, v' ]
        TF.gradients y [v,v'] >>= TF.run
    V.fromList [1 :: Float] @=? dv    
    V.fromList [1 :: Float] @=? dv'
    [dv,dv'] <- TF.runSession $ do
        v  <- TF.render $ TF.vector [1,2,3,4 :: Float]
        v' <- TF.render $ TF.vector [5,6,7,8 :: Float]
        let y = TF.concat (TF.scalar 0) [ v, v', v ]
        TF.gradients y [v,v'] >>= TF.run
    V.fromList [2,2,2,2 :: Float] @=? dv    
    V.fromList [1,1,1,1 :: Float] @=? dv'

-- TODO(JAK): - remove _foldl 
--            - hlint
--            - remove rnd
-- TODO(JAK): description ...
-- This test checks that ...
--   similar to
-- tensorflow/tensorflow/compiler/tests/concat_ops_test.py 
--  ConcatTest._testGradientsSimple
testConcatGradientSimple :: Test
testConcatGradientSimple = testCase "testConcatGradientSimple" $ do
    let shapes     = [[10,x,2] | x <- [1,2,6]]
        _foldl f (x:xs) = foldl f x xs
    (inputGrads :: [[Float]]) <- forM shapes $ \shape ->
       replicateM (List.product shape) randomIO
    (inputs :: [[Float]]) <- forM shapes $ \shape ->
       replicateM (List.product shape) randomIO
    dinputs <- TF.runSession $ do
        inputTensors <- forM (inputs `zip` shapes) $ \(input,shape) -> 
                          TF.render $ TF.constant (TF.Shape shape) input
        inputGradTensors <- forM (inputGrads `zip` shapes) $ \(inputGrad, shape) -> 
                               TF.render $ TF.constant (TF.Shape shape) inputGrad
        inputGradTensor <- TF.render $ TF.concat (TF.scalar 1) inputGradTensors
        inputTensor <- TF.render $ TF.concat (TF.scalar 1) inputTensors
        output <- TF.render $ inputTensor `TF.mul` inputGradTensor
        TF.gradients output inputTensors >>= TF.run
    (V.fromList <$> inputGrads) @=? dinputs

-- TODO(JAK): description ...
-- This test checks that ...
--   similar to
-- tensorflow/tensorflow/compiler/tests/concat_ops_test.py 
--  ConcatTest._testGradientsSimple
testConcatGradientFirstDim :: Test
testConcatGradientFirstDim = testCase "testConcatGradientFirstDim" $ do
    let shapes     = [[x,10,2] | x <- [1,2,6]]
        _foldl f (x:xs) = foldl f x xs
    (inputGrads :: [[Float]]) <- forM shapes $ \shape ->
       replicateM (List.product shape) randomIO
    (inputs :: [[Float]]) <- forM shapes $ \shape ->
       replicateM (List.product shape) randomIO
    dinputs <- TF.runSession $ do
        inputTensors <- forM (inputs `zip` shapes) $ \(input,shape) -> 
                          TF.render $ TF.constant (TF.Shape shape) input
        inputGradTensors <- forM (inputGrads `zip` shapes) $ \(inputGrad, shape) -> 
                               TF.render $ TF.constant (TF.Shape shape) inputGrad
        inputGradTensor <- TF.render $ TF.concat (TF.scalar 0) inputGradTensors
        inputTensor <- TF.render $ TF.concat (TF.scalar 0) inputTensors
        output <- TF.render $ inputTensor `TF.mul` inputGradTensor
        TF.gradients output inputTensors >>= TF.run
    (V.fromList <$> inputGrads) @=? dinputs

-- TODO(JAK): description ...
-- This test checks that ...
--   similar to
-- tensorflow/tensorflow/compiler/tests/concat_ops_test.py 
--  ConcatTest._testGradientsSimple
testConcatGradientLastDim :: Test
testConcatGradientLastDim = testCase "testConcatGradientLastDim" $ do
    let shapes     = [[10,2,x] | x <- [1,2,6]]
        _foldl f (x:xs) = foldl f x xs
    (inputGrads :: [[Float]]) <- forM shapes $ \shape ->
       replicateM (List.product shape) randomIO
    (inputs :: [[Float]]) <- forM shapes $ \shape ->
       replicateM (List.product shape) randomIO
    dinputs <- TF.runSession $ do
        inputTensors <- forM (inputs `zip` shapes) $ \(input,shape) -> 
                          TF.render $ TF.constant (TF.Shape shape) input
        inputGradTensors <- forM (inputGrads `zip` shapes) $ \(inputGrad, shape) -> 
                               TF.render $ TF.constant (TF.Shape shape) inputGrad
        inputGradTensor <- TF.render $ TF.concat (TF.scalar 2) inputGradTensors
        inputTensor <- TF.render $ TF.concat (TF.scalar 2) inputTensors
        output <- TF.render $ inputTensor `TF.mul` inputGradTensor
        TF.gradients output inputTensors >>= TF.run
    (V.fromList <$> inputGrads) @=? dinputs


testConcatRunAndVerifyGradientsRandom :: Test
testConcatRunAndVerifyGradientsRandom = testCase "testConcatRunAndVerifyGradientsRandom" $ 
    forM_ [1..5] $ \_ -> do
      let rnd _min _max =  ((+ _min) . (`mod` (_max - _min))) <$> randomIO
      (shapes' :: [Int64]) <- replicateM 5 $ rnd 1 5
      (numTensors :: Int) <- rnd 1 10
      (concatDim :: Int32) <- rnd 0 4
      (concatDimSizes :: [Int64]) <- replicateM numTensors $ rnd 1 5
      let concatDimSize = sum concatDimSizes
          update i xs x = xs `go` [0..]
            where go [] _ = []
                  go (y:ys) (j:js) | i == j    = x:ys
                                   | otherwise = y:go ys js
          shapes = map (update concatDim shapes') concatDimSizes
          wholeshape = update concatDim shapes' concatDimSize
      print concatDim
      print numTensors
      forM_ shapes print 
      let _foldl f (x:xs) = foldl f x xs
      (inputGrads :: [[Float]]) <- forM shapes $ \shape ->
         replicateM (fromIntegral $ List.product shape) randomIO
      (inputs :: [[Float]]) <- forM shapes $ \shape ->
         replicateM (fromIntegral $ List.product shape) randomIO
      dinputs <- TF.runSession $ do
          inputTensors <- forM (inputs `zip` shapes) $ \(input,shape) -> 
                            TF.render $ TF.constant (TF.Shape shape) input
          inputTensor <- TF.render $ TF.concat (TF.scalar concatDim) inputTensors
          inputGradTensors <- forM (inputGrads `zip` shapes) $ \(inputGrad, shape) -> 
                                 TF.render $ TF.constant (TF.Shape shape) inputGrad
          inputGradTensor <- TF.render $ TF.concat (TF.scalar concatDim) inputGradTensors
          output <- TF.render $ inputTensor `TF.mul` inputGradTensor
          TF.gradients output inputTensors >>= TF.run
      (V.fromList <$> inputGrads) @=? dinputs

main :: IO ()
main = googleTest [ testGradientSimple
                  , testGradientDisconnected
                  , testCreateGraphStateful
                  , testCreateGraphNameScopes
                  , testDiamond
                  , testMaxGradient
                  , testConcatGradient
                  , testConcatGradientSimple
                  , testConcatGradientFirstDim
                  , testConcatGradientLastDim
                  , testConcatRunAndVerifyGradientsRandom
                  ]

--
--  def _RunAndVerifyGradientsRandom(self):
--    # Random dims of rank 5
--    input_shape = np.random.randint(1, 5, size=5)
--    # Random number of tensors
--    num_tensors = np.random.randint(1, 10)
--    # Random dim to concat on
--    concat_dim = np.random.randint(5)
--    concat_dim_sizes = np.random.randint(1, 5, size=num_tensors)
--    with self.test_session():
--      inp = []
--      inp_tensors = []
--      with self.test_scope():
--        for x in concat_dim_sizes:
--          shape = input_shape
--          shape[concat_dim] = x
--          t = np.random.rand(*shape).astype("f")
--          inp.append(t)
--          inp_tensors.append(
--              constant_op.constant(
--                  [float(y) for y in t.flatten()],
--                  shape=shape,
--                  dtype=dtypes.float32))
--        c = array_ops.concat(inp_tensors, concat_dim)
--        output_shape = input_shape
--        output_shape[concat_dim] = concat_dim_sizes.sum()
--        grad_inp = np.random.rand(*output_shape).astype("f")
--        grad_tensor = constant_op.constant(
--            [float(x) for x in grad_inp.flatten()], shape=output_shape)
--        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
--        concated_grad = array_ops.concat(grad, concat_dim)
--        result = concated_grad.eval()
--
--    self.assertAllEqual(result, grad_inp)
--
--
--  # Re-enable once zero-element Retvals are handled correctly.
--  def DISABLED_testZeroSize(self):
--    # Verify that concat doesn't crash and burn for zero size inputs
--    np.random.seed(7)
--    with self.test_session() as sess:
--      with self.test_scope():
--        for shape0 in (), (2,):
--          axis = len(shape0)
--          for shape1 in (), (3,):
--            for n0 in 0, 1, 2:
--              for n1 in 0, 1, 2:
--                x0 = np.random.randn(*(shape0 + (n0,) + shape1))
--                x1 = np.random.randn(*(shape0 + (n1,) + shape1))
--                correct = np.concatenate([x0, x1], axis=axis)
--                # TODO(irving): Make tf.concat handle map, then drop list().
--                xs = list(map(constant_op.constant, [x0, x1]))
--                c = array_ops.concat(xs, axis)
--                self.assertAllEqual(c.eval(), correct)
--                # Check gradients
--                dc = np.random.randn(*c.get_shape().as_list())
--                dxs = sess.run(gradients_impl.gradients(c, xs, dc))
--                self.assertAllEqual(dc, np.concatenate(dxs, axis=axis))


