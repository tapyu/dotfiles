classdef Opset12TranslationStrategy < nnet.internal.cnn.onnx.fcn.matlab.Opset11TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 12.

    %   Copyright 2020-2023 The MathWorks, Inc.
    methods
        nodeTranslation = translateArgMax(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateConstant(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateDropout(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGatherND(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateLessOrEqual(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGreaterOrEqual(this, nodeTranslation, node, IntegerTensorNames);
    end
end
