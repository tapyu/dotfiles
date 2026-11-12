classdef Opset20TranslationStrategy < nnet.internal.cnn.onnx.fcn.matlab.Opset19TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 20.

    %   Copyright 2024 The MathWorks, Inc.
    methods
        nodeTranslation = translateGelu(~, nodeTranslation, node, ~);
        nodeTranslation = translateGridSample(~, nodeTranslation, node, IntegerTensorNames);
    end
end