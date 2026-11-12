classdef Opset13TranslationStrategy < nnet.internal.cnn.onnx.fcn.matlab.Opset12TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 13.

    %   Copyright 2020-2025 The MathWorks, Inc.
    methods
        % Translation utilities 
        inputStr = iGetOptionalInputStr(this, node, inputIdx);

        % Translators
        nodeTranslation = translateHardmax(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateLogSoftmax(~, nodeTranslation, node, ~);
        nodeTranslation = translateReduceSum(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateResize(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSoftmax(~, nodeTranslation, node, ~);
        nodeTranslation = translateSplit(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSqueeze(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateUnsqueeze(~, nodeTranslation, node, IntegerTensorNames);
    end
end
