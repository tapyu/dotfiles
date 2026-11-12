classdef ConverterForLayerNormalizationLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a LayerNormalizationLayer into ONNX
    
    % Copyright 2024 The MathWorks, Inc.
    
    methods
        function this = ConverterForLayerNormalizationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames            = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout           = TensorLayoutMap(inputTensorNames{1});
            nextOperatorInput = this.InputLayerNames{1};

            iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:LayerNormalizationLayerFixedInputSize', this.NNTLayer.Name);
            
            %  Permute input dimension corresponding to the
            %  OperationDimension Attribute
            switch this.NNTLayer.OperationDimension
                case "auto"
                    if find(char(inputTensorLayout)=='w')
                        [axis, normDim, permutation, reversePermutation] = getSpatialChannelPermutationAndAxis(this,inputTensorLayout);
                    else
                        [axis, normDim, permutation, reversePermutation] = getChannelOnlyPermutationAndAxis(this,inputTensorLayout);
                    end
                case "channel-only"
                    [axis, normDim, permutation, reversePermutation] = getChannelOnlyPermutationAndAxis(this,inputTensorLayout);
                case "spatial-channel"
                    [axis, normDim, permutation, reversePermutation] = getSpatialChannelPermutationAndAxis(this,inputTensorLayout);
                case "batch-excluded"
                    [axis, normDim, permutation, reversePermutation] = getBatchExcludedPermutationAndAxis(this,inputTensorLayout);
            end

            if ~isempty(permutation)
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, permutation);
            end

            % Make the nodeProto
            layerNormNode           = NodeProto;
            layerNormNode.op_type	= 'LayerNormalization';
            layerNormNode.name      = onnxName;
            layerNormNode.input     = mapTensorNames(this,{nextOperatorInput, [onnxName '_scale'], [onnxName '_B']}, TensorNameMap);
            layerNormNode.output    = {onnxName};
            layerNormNode.attribute = [...
                makeAttributeProto('epsilon', 'FLOAT', this.NNTLayer.Epsilon),...
                makeAttributeProto('axis', 'INT', axis),...
                makeAttributeProto('stash_type','INT',0)];

            % ONNX LayerNormalization has element-wise scale and offset.
            % Expand scale and offset to element-wise. 
            normDimSize = this.FullInputSizes{1}(normDim);          % dimension size to normalize in forward-ONNX format
            
            % Make parameter Initializers for: scale, B
            scale = reshape(this.NNTLayer.Scale,[],1);
            if size(normDimSize,2)>1
                scale = repmat(scale,[1 normDimSize(2:end)]);
            end
            t1 = TensorProto;
            t1.name = [onnxName '_scale'];
            t1.data_type = TensorProto_DataType.FLOAT;
            t1.raw_data = rawData(single(scale));
            t1.dims = dimVector(size(scale),ndims(scale));            
            offset = reshape(this.NNTLayer.Offset,[],1);
            if size(normDimSize,2)>1
                offset = repmat(offset,[1 normDimSize(2:end)]);
            end
            t2 = TensorProto;
            t2.name = [onnxName '_B'];
            t2.data_type = TensorProto_DataType.FLOAT;
            t2.raw_data = rawData(single(offset));
            t2.dims = dimVector(size(offset),ndims(offset));
            
            parameterInitializers = [t1 t2];
            
            nodeProto(end+1)        = layerNormNode;
            networkInputs           = [];
            networkOutputs          = [];
            
            nextOperatorInput       = onnxName;
            
            % Permute back to reverse MATLAB format
            if ~isempty(reversePermutation)
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, reversePermutation);
            end

            % Update maps
            TensorNameMap(this.NNTLayer.Name) = nextOperatorInput;
            TensorLayoutMap(nextOperatorInput) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
    methods (Access=private)
        function [axis, normDim, permutation, reversePermutation] = getSpatialChannelPermutationAndAxis(~,inputTensorLayout)
            permutation = [];
            reversePermutation = [];
            axis = 1;
            switch inputTensorLayout
                case {'snc','snch','snchw','snchwd'}
                    axis = 2;
                case {'ch','chw','chwd'}
                    axis = 0;
            end
            normDim = (strlength(inputTensorLayout)-axis):-1:1;
        end

        function [axis, normDim, permutation, reversePermutation] = getChannelOnlyPermutationAndAxis(~,inputTensorLayout)
            axis = -1;
            normDim = strlength(inputTensorLayout) + 1 - find(char(inputTensorLayout)=='c');
            permutation = [];
            reversePermutation = [];
            switch inputTensorLayout
                case {'sch','nch'}
                    permutation = [0,2,1];
                    reversePermutation = [0,2,1];
                case {'schw', 'nchw'}
                    permutation = [0,2,3,1];
                    reversePermutation = [0,3,1,2];
                case {'schwd', 'nchwd'}
                    permutation = [0,2,3,4,1];
                    reversePermutation = [0,4,1,2,3];
                case 'snch'
                    permutation = [0,1,3,2];
                    reversePermutation = [0,1,3,2];
                case 'snchw'
                    permutation = [0,1,3,4,2];
                    reversePermutation = [0,1,4,2,3];
                case 'snchwd'
                    permutation = [0,1,3,4,5,2];
                    reversePermutation = [0,1,5,2,3,4];
                case 'ch'
                    permutation = [1,0];
                    reversePermutation = [1,0];
                case 'chw'
                    permutation = [1,2,0];
                    reversePermutation = [2,0,1];
                case 'chwd'
                    permutation = [1,2,3,0];
                    reversePermutation = [3,0,1,2];
            end
        end

        function [axis, normDim, permutation, reversePermutation] = getBatchExcludedPermutationAndAxis(~,inputTensorLayout)
            permutation = [];
            reversePermutation = [];
            if isempty(find(char(inputTensorLayout)=='n',1))
                axis = 0;
            else
                axis = 1;
            end
            cDim = find(char(inputTensorLayout)=='c');
            sDim = find(char(inputTensorLayout)=='s');
            hDim = find(char(inputTensorLayout)=='h');
            wDim = find(char(inputTensorLayout)=='w');
            dDim = find(char(inputTensorLayout)=='d');
            normDim = strlength(inputTensorLayout) + 1 - [cDim, sDim, hDim, wDim, dDim];
            switch inputTensorLayout
                case 'sc'
                    permutation = [1,0];
                    reversePermutation = permutation;
                case 'sch'
                    permutation = [1,0,2];
                    reversePermutation = permutation;
                case 'schw'
                    permutation = [1,0,2,3];
                    reversePermutation = permutation;
                case 'schwd'
                    permutation = [1,0,2,3,4];
                    reversePermutation = permutation;
                case 'snc'
                    permutation = [1,2,0];
                    reversePermutation = [2,0,1];
                case 'snch'
                    permutation = [1,2,0,3];
                    reversePermutation = [2,0,1,3];
                case 'snchw'
                    permutation = [1,2,0,3,4];
                    reversePermutation = [2,0,1,3,4];
                case 'snchwd'
                    permutation = [1,2,0,3,4,5];
                    reversePermutation = [2,0,1,3,4,5];
            end
        end
    end
end

function iWarningWithoutBacktrace(msgID, varargin)
    backtrace = warning('query','backtrace');
    warning('off','backtrace');
    warning(message(msgID, varargin{:}));
    warning(backtrace.state,'backtrace');
end
