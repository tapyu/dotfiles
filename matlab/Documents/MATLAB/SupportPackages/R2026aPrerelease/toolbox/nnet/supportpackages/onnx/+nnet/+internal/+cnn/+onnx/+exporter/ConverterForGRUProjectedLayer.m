classdef ConverterForGRUProjectedLayer < nnet.internal.cnn.onnx.exporter.ConverterForGRULayer
    % Class to convert a gruProjectedLayer into gruLayer in ONNX

    % Supported input formats: S*CB[T] for the main input and CB for state inputs. 

    % Copyright 2023-2025 The MathWorks, Inc.

    methods
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            inputWeights = this.NNTLayer.InputWeights * this.NNTLayer.InputProjector';
            recurrentWeights = this.NNTLayer.RecurrentWeights * this.NNTLayer.OutputProjector';
            % Reuse the inherited method convertGRUWithWeightsToONNX.
            [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap]...
                = this.convertGRUWithWeightsToONNX(nodeProto, TensorNameMap, TensorLayoutMap, ...
                inputWeights, recurrentWeights);
        end
    end
end    