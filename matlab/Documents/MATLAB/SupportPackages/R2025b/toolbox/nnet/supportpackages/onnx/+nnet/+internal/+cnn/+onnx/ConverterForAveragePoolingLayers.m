classdef ConverterForAveragePoolingLayers < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an average pooling layer (averagePooling2dLayer or
    % averagePooling3dLayer) to ONNX
    
    % Copyright 2018-2022 The MathWorks, Inc.
    
    methods
        function this = ConverterForAveragePoolingLayers(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
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
            %  AvgPool to operate over TBC -> BCT
            if isequal(inputTensorLayout,'snc')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 0]);
            %  AvgPool to operate over TBCS -> BCST
            elseif isequal(inputTensorLayout,'snch')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 3 0]);
            %  AvgPool to operate over TBCSS -> BCSST
            elseif isequal(inputTensorLayout,'snchw')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 3 4 0]);
            end

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
            
            % Create the avgpool node
            avgPoolNode           = NodeProto;
            avgPoolNode.op_type   = 'AveragePool';
            avgPoolNode.name      = onnxName;
            avgPoolNode.input     = mapTensorNames(this, {nextOperatorInput}, TensorNameMap);
            avgPoolNode.output    = {onnxName};
            avgPoolNode.attribute = [...
                makeAttributeProto('kernel_shape', 'INTS', poolSize),...
                makeAttributeProto('pads',         'INTS', paddingSize),...
                makeAttributeProto('strides',      'INTS', stride)
                ];
            if this.OpsetVersion >= 7
                if this.NNTLayer.PaddingValue == 0
                    avgPoolNode.attribute(end+1) = makeAttributeProto('count_include_pad', 'INT', 1);
                else
                    avgPoolNode.attribute(end+1) = makeAttributeProto('count_include_pad', 'INT', 0);
                end
            end
            
            nextOperatorInput       = onnxName;
            nodeProto(end+1)        = avgPoolNode;

            %Output a sequence if the input was a sequence by permuting
            %the sequence dimension -> BCT to TBC
            if isequal(inputTensorLayout,'snc')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [2 0 1]);
            %the sequence dimension -> BCST to TBCS
            elseif isequal(inputTensorLayout,'snch')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [3 0 1 2]);
            %the sequence dimension -> BCSST to TBCSS
            elseif isequal(inputTensorLayout,'snchw')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [4 0 1 2 3]);
            end

            parameterInitializers   = [];
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
