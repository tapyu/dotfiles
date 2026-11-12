classdef Opset18TranslationStrategy < nnet.internal.cnn.onnx.fcn.coder.Opset17TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 18.

    %   Copyright 2024 The MathWorks, Inc.
    methods
        nodeTranslation = translatePad(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSplit(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateResize(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceL1(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceL2(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceLogSum(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceLogSumExp(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceMax(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceMean(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceMin(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceProd(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceSumSquare(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceOp(~, nodeTranslation, node, IntegerTensorNames, op);
        
    end
end