classdef ConverterForGlobalMaxPooling1dLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a builtin globalMaxPooling1DLayer into ONNX
        
    % Copyright 2021-2025 The MathWorks, Inc.

    methods
        function this = ConverterForGlobalMaxPooling1dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            nextOperatorInput = this.InputLayerNames{1};
            
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            [nodeProto, nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, 1);

            if ismember(inputTensorLayout,{'snc','nch'})
                % GMP node
                [gmpName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
                gmpName                = makeUniqueName({nodeProto.name}, gmpName);
                [gmpNode, gmpInit]  = createNodeProto(this, 'SISO', 'GlobalMaxPool', gmpName, ...
                    mapTensorNames(this, {nextOperatorInput}, TensorNameMap), {gmpName});
                nodeProto(end+1)        = gmpNode;
                parameterInitializers   = gmpInit;
                networkInputs           = [];
                networkOutputs          = [];
                nextOperatorInput       = gmpName;
                outputTensorLayout      = 'nch';
                % squeeze node
                if isequal(inputTensorLayout,'snc')
                    %Squeeze nc1 to nc
                    squeezeName                 = [gmpName '_Squeeze'];
                    squeezeName                 = makeUniqueName([{nodeProto.name}, {nextOperatorInput}], squeezeName);
                    squeezeInput                = {gmpName};
                    squeezeOutput               = {squeezeName};
                    [squeezeNode, squeezeInit]  = createNodeProto(this, 'Squeeze', squeezeName, squeezeInput, squeezeOutput, 2);                                        
                    nodeProto(end+1)            = squeezeNode;
                    parameterInitializers       = [parameterInitializers squeezeInit];                   

                    nextOperatorInput      = squeezeName;
                    outputTensorLayout     = 'nc';
                end

                % Update maps
                TensorNameMap(this.NNTLayer.Name)     = nextOperatorInput;
                TensorLayoutMap(nextOperatorInput)    = outputTensorLayout;
            elseif isequal(inputTensorLayout,'snch')
                % GMP node
                [gmpName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
                gmpName                = makeUniqueName({nodeProto.name}, gmpName);
                [gmpNode, gmpInit]  = createNodeProto(this, 'MaxPool', gmpName, ...
                    mapTensorNames(this, {nextOperatorInput}, TensorNameMap), {gmpName}, ...
                    [this.InputLayerSizes{1}(1) 1], [], []);                                        

                nodeProto(end+1)        = gmpNode;
                parameterInitializers   = gmpInit;
                networkInputs           = [];
                networkOutputs          = [];
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, gmpName, TensorNameMap, [3 0 1 2]);

                % Update maps
                TensorNameMap(this.NNTLayer.Name)           = nextOperatorInput;
                TensorLayoutMap(nextOperatorInput)          = 'snch';
            end

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end