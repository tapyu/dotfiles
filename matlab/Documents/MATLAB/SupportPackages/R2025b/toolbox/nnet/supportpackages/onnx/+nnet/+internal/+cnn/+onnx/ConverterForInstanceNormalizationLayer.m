classdef ConverterForInstanceNormalizationLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a instanceNormalizationLayer into ONNX

    % Copyright 2020-2024 The MathWorks, Inc.

    methods
        function this = ConverterForInstanceNormalizationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
            parameterInitializers   = [];

            % Make the InstanceNormalization NodeProto
            instanceNormNode           = NodeProto;
            instanceNormNode.op_type   = 'InstanceNormalization';
            instanceNormNode.name      = onnxName;
            instanceNormNode.output    = {onnxName};
            instanceNormNode.attribute = makeAttributeProto('epsilon', 'FLOAT', this.NNTLayer.Epsilon);

            if isequal(inputTensorLayout, '1nc')
                % InstanceNormalization requires its input to be in the
                % form B,C,S...S, or BC if there are no spatial dimensions.
                % Squeeze out the leading singleton dim.
                squeezeName                 = [onnxName '_squeeze'];
                squeezeInput                = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
                squeezeOutput               = {[onnxName '_squeeze']};
                [squeezeNode, squeezeInit]  = createNodeProto(this, 'Squeeze', squeezeName, squeezeInput, squeezeOutput, 0);
                parameterInitializers       = [parameterInitializers squeezeInit];

                instanceNormNode.input  = mapTensorNames(this,{[onnxName '_squeeze'], [onnxName '_scale'], [onnxName '_B']}, TensorNameMap);
                nodeProto(end+1:end+2)  = [squeezeNode, instanceNormNode];

                % Set the output tensor format to NC
                TensorLayoutMap(onnxName) = 'nc';

            else
                instanceNormNode.input  = mapTensorNames(this,{this.InputLayerNames{1}, [onnxName '_scale'], [onnxName '_B']}, TensorNameMap);
                nodeProto(end+1)        = instanceNormNode;

                % Set the output tensor format to match the input
                TensorLayoutMap(onnxName) = inputTensorLayout;
            end

            % Make parameter Initializers for: scale, B
            t1 = TensorProto;
            t1.name = [onnxName '_scale'];
            t1.data_type = TensorProto_DataType.FLOAT;
            t1.raw_data = rawData(single(this.NNTLayer.Scale));
            t1.dims = dimVector(numel(this.NNTLayer.Scale),1);           % NNT data: 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)

            t2 = TensorProto;
            t2.name = [onnxName '_B'];
            t2.data_type = TensorProto_DataType.FLOAT;
            t2.raw_data = rawData(single(this.NNTLayer.Offset));
            t2.dims = dimVector(numel(this.NNTLayer.Offset),1);          % NNT data: 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)

            parameterInitializers = [parameterInitializers t1 t2];

            % Make network Inputs
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            TensorNameMap(this.NNTLayer.Name) = onnxName;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
