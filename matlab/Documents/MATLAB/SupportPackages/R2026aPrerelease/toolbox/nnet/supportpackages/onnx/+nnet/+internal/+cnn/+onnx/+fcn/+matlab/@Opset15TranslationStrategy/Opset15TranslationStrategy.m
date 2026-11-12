classdef Opset15TranslationStrategy < nnet.internal.cnn.onnx.fcn.matlab.Opset14TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 15.

    %   Copyright 2024 The MathWorks, Inc.
    methods
        % Translators
        nodeTranslation = translateShape(~, nodeTranslation, node, ~);
        nodeTranslation = translateBernoulli(~, nodeTranslation, node, ~);
    end
end
