classdef ConverterForLSTMLayer < nnet.internal.cnn.onnx.ConverterForLSTMLayer_Base
    % Class to convert a lstmLayer into ONNX

    % Supported IO as of 21b:
    % CBT(sequence) --> CBT
    % CBT(last) --> CB
    % Hidden and Cell inputs and outputs, when present, are always CB.

    % Copyright 2018-2022 The MathWorks, Inc.

    methods
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap]...
                = toOnnx@nnet.internal.cnn.onnx.ConverterForLSTMLayer_Base(this, nodeProto, TensorNameMap, TensorLayoutMap, ...
                this.NNTLayer.InputWeights, this.NNTLayer.RecurrentWeights);
        end
    end
end