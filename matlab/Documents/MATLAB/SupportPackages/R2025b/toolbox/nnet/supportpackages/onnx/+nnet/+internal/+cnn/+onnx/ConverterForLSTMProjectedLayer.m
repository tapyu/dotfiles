classdef ConverterForLSTMProjectedLayer < nnet.internal.cnn.onnx.ConverterForLSTMLayer_Base
    % Class to convert a lstmProjectedLayer into lstmLayer in ONNX

    % Supported input formats: S*CB[T] for the main input and CB for state inputs. 

    % Copyright 2022-2023 The MathWorks, Inc.

    methods
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            inputWeights = this.NNTLayer.InputWeights * this.NNTLayer.InputProjector';
            recurrentWeights = this.NNTLayer.RecurrentWeights * this.NNTLayer.OutputProjector';
            [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap]...
                = toOnnx@nnet.internal.cnn.onnx.ConverterForLSTMLayer_Base(this, nodeProto, TensorNameMap, TensorLayoutMap, ...
                inputWeights, recurrentWeights);
        end
    end
end    