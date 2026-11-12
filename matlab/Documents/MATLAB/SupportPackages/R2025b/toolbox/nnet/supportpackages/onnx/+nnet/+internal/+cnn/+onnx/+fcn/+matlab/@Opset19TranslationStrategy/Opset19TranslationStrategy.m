classdef Opset19TranslationStrategy < nnet.internal.cnn.onnx.fcn.matlab.Opset18TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 19.

    %   Copyright 2024 The MathWorks, Inc.    
    methods
        nodeTranslation = translateAveragePool(~, nodeTranslation, node, ~)
    end
end