classdef ConverterForGlobalAveragePooling3dLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a builtin globalAveragePooling3DLayer into ONNX
        
    % Copyright 2022-2025 The MathWorks, Inc.

    methods
        function this = ConverterForGlobalAveragePooling3dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            nextOperatorInput = this.InputLayerNames{1};
            
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            % Exporting pooling/convolution layers operating over SSSCBT data is
            % currently unsupported
            if isequal(inputTensorLayout,'snchwd')
                error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end
            [nodeProto, nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, 1);

            if ismember(inputTensorLayout,{'snchw','nchwd'})
                
                % GAP node
                [gapName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
                gapName                = makeUniqueName({nodeProto.name}, gapName);
                gapNode = createNodeProto(this, 'SISO', 'GlobalAveragePool', gapName,...
                    mapTensorNames(this, {nextOperatorInput}, TensorNameMap), {gapName});
                
                nodeProto(end+1)        = gapNode;
                parameterInitializers   = [];
                networkInputs           = [];
                networkOutputs          = [];

                nextOperatorInput       = gapName;
                outputTensorLayout      = 'nchwd';
                if isequal(inputTensorLayout,'snchw')
                    %Squeeze nch1 to nch
                    squeezeName                 = [gapName '_Squeeze'];
                    squeezeName                 = makeUniqueName([{nodeProto.name}, {nextOperatorInput}], squeezeName);
                    squeezeInput                = {gapName};
                    squeezeOutput               = {squeezeName};
                    [squeezeNode, squeezeInit]  = createNodeProto(this, 'Squeeze', squeezeName, squeezeInput, squeezeOutput, 4);                    
                    nodeProto(end+1)            = squeezeNode;
                    parameterInitializers       = [parameterInitializers squeezeInit];
                    
                    nextOperatorInput      =  squeezeName;
                    outputTensorLayout     = 'nchw';
                end
                % Update maps
                TensorNameMap(this.NNTLayer.Name) = nextOperatorInput;
                TensorLayoutMap(nextOperatorInput)= outputTensorLayout;
            end
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
