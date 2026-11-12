classdef ConverterForAveragePoolingLayers < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert an average pooling layer (averagePooling2dLayer or
    % averagePooling3dLayer) to ONNX
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForAveragePoolingLayers(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});

            % Exporting average pooling layers operating over SSSCBT data
            % is currently unsupported
            if isequal(inputTensorLayout,'snchwd')
                error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end

            nextOperatorInput = this.InputLayerNames{1};
            %  Permute Sequence input dimension to Spatial Dimension for
            %  AvgPool. Permute from the first dim to the last dim
            [nodeProto, nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, 1);
            
            nntLayer                = this.NNTLayer;
            [onnxName, ~]           = legalizeNNTName(this, nntLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);

            % The layer can be either an averagePooling1dLayer, averagePooling2dLayer or
            % averagePooling3dlayer. Determine what shape parameters to use
            outputTensorLayout = inputTensorLayout;
            if isa(this.NNTLayer, 'nnet.cnn.layer.AveragePooling1DLayer')
                 if ismember(inputTensorLayout,{'snc','nch'})
                     poolSize           = nntLayer.PoolSize;
                     paddingIdx         = [1 2]; % NNT is [left right], ONNX is [left right]
                     paddingSize        = nntLayer.PaddingSize(paddingIdx);
                     stride             = nntLayer.Stride;
                 elseif ismember(inputTensorLayout,{'snch'})
                     poolSize           = [nntLayer.PoolSize 1]; %Kernels of size 1 indicate no-ops over sequence dimension
                     paddingSize        = [nntLayer.PaddingSize(1) 0 nntLayer.PaddingSize(2) 0];
                     stride             = [nntLayer.Stride 1];
                 end
            elseif isa(this.NNTLayer, 'nnet.cnn.layer.AveragePooling2DLayer')
                if ismember(inputTensorLayout,{'nchw','snch'})
                    poolSize            = nntLayer.PoolSize;
                    paddingIdx          = [1 3 2 4]; % NNT is [top bottom left right], ONNX is [top left bottom right]
                    paddingSize         = nntLayer.PaddingSize(paddingIdx);
                    stride              = nntLayer.Stride;
                elseif ismember(inputTensorLayout,{'snchw'})
                    poolSize           = [nntLayer.PoolSize 1]; %Kernels of size 1 indicate no-ops over sequence dimension
                    paddingSize        = [nntLayer.PaddingSize(1) nntLayer.PaddingSize(3) 0 nntLayer.PaddingSize(2) nntLayer.PaddingSize(4) 0];
                    stride             = [nntLayer.Stride 1];
                end
            elseif isa(this.NNTLayer, 'nnet.cnn.layer.AveragePooling3DLayer')
                poolSize            = nntLayer.PoolSize;
                paddingIdx          = [1 3 5 2 4 6]; % NNT is [top bottom left right front back], ONNX is [top left front bottom right back]
                paddingSize         = nntLayer.PaddingSize(paddingIdx);
                stride              = nntLayer.Stride;  
            else
                error(message("nnet_cnn_onnx:onnx:UnexpectedLayer", 'nnet.cnn.layer.AveragePooling2DLayer', 'nnet.cnn.layer.AveragePooling3DLayer', class(this.NNTLayer)));
            end
            count_include_pad = [];
            if this.OpsetVersion >= 7
                if this.NNTLayer.PaddingValue == 0
                    count_include_pad = 1;
                else
                    count_include_pad = 0;
                end
            end

            % Create the avgpool node
            [avgPoolNode, parameterInitializers] = createNodeProto(this, 'AveragePool', onnxName, ...
                mapTensorNames(this, {nextOperatorInput}, TensorNameMap),...
                {onnxName}, poolSize, paddingSize, stride, count_include_pad);
            
            nextOperatorInput       = onnxName;
            nodeProto(end+1)        = avgPoolNode;

            %Output a sequence if the input was a sequence by permuting
            %the sequence dimension from the last dim to the first dim
            [nodeProto, nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, 0);

            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            TensorNameMap(this.NNTLayer.Name)   = nextOperatorInput;
            TensorLayoutMap(nextOperatorInput)  = outputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
