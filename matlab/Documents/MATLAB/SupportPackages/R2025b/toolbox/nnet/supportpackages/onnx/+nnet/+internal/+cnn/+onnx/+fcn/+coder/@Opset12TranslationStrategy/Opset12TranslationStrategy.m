classdef Opset12TranslationStrategy < nnet.internal.cnn.onnx.fcn.coder.Opset11TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 12.

    %   Copyright 2024 The MathWorks, Inc.
    methods
        nodeTranslation = translateArgMax(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateConstant(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateDropout(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGatherND(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateLessOrEqual(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGreaterOrEqual(this, nodeTranslation, node, IntegerTensorNames);
    end
end
