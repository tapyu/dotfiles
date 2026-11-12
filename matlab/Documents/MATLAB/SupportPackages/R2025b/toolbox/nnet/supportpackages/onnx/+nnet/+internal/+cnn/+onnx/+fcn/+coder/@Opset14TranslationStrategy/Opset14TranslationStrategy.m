classdef Opset14TranslationStrategy < nnet.internal.cnn.onnx.fcn.coder.Opset13TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 14.

    %   Copyright 2024 The MathWorks, Inc.
    methods
        
        % Translation utilities 
        C = iStringifyEmpties(this, C);
        s = iTranslateActivationFunction(this, s);
        
        % Translators
        nodeTranslation = translateGRU(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateLSTM(this, nodeTranslation, node, ~);
        nodeTranslation = translateReshape(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateHardSwish(this, nodeTranslation, node, ~);
    end
end