classdef Opset16TranslationStrategy < nnet.internal.cnn.onnx.fcn.coder.Opset15TranslationStrategy
%

%   Copyright 2024 The MathWorks, Inc.
    methods
        % Translators
        nodeTranslation = translateGridSample(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateScatterElements(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateScatterND(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateRoiAlign(~, nodeTranslation, node, IntegerTensorNames);
    end
end