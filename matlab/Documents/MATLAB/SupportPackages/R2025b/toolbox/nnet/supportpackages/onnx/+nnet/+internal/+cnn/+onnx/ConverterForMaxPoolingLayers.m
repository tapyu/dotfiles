classdef ConverterForMaxPoolingLayers < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a MaxPooling2dLayer or MaxPooling3dLayer into ONNX
    
    % Copyright 2018-2022 The MathWorks, Inc.
    
    methods
        function this = ConverterForMaxPoolingLayers(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});

            % Exporting max pooling layers operating over SSSCBT data is
            % currently unsupported
            if isequal(inputTensorLayout,'snchwd')
                error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end

            nextOperatorInput = this.InputLayerNames{1};

            %  Permute Sequence input dimension to Spatial Dimension for
            %  MaxPool to operate over TBC -> BCT
            if isequal(inputTensorLayout,'snc')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 0]);
            %  MaxPool to operate over TBCS -> BCST
            elseif isequal(inputTensorLayout,'snch')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 3 0]);
            %  MaxPool to operate over TBCSS -> BCSST
            elseif isequal(inputTensorLayout,'snchw')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 3 4 0]);
            end

            nntLayer                = this.NNTLayer;
            [onnxName, ~]           = legalizeNNTName(this, nntLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
                        
            % The layer can be a maxPooling1dLayer, maxPooling2dLayer or
            % maxPooling3dlayer. Determine the shape of the parameters.
            outputTensorLayout = inputTensorLayout;
            if isa(this.NNTLayer, 'nnet.cnn.layer.MaxPooling1DLayer')
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
            elseif isa(this.NNTLayer, 'nnet.cnn.layer.MaxPooling2DLayer')
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
            elseif isa(this.NNTLayer, 'nnet.cnn.layer.MaxPooling3DLayer')
                poolSize            = nntLayer.PoolSize;
                paddingIdx          = [1 3 5 2 4 6]; % NNT is [top bottom left right front back], ONNX is [top left front bottom right back]
                paddingSize         = nntLayer.PaddingSize(paddingIdx);
                stride              = nntLayer.Stride;  
            else
                error(message("nnet_cnn_onnx:onnx:UnexpectedLayer", 'nnet.cnn.layer.MaxPooling2dLayer', 'nnet.cnn.layer.MaxPooling3dLayer', class(this.NNTLayer)));
            end            
            
            % Create the maxpool node
            maxPoolNode           = NodeProto;
            maxPoolNode.op_type   = 'MaxPool';
            maxPoolNode.name      = onnxName;
            maxPoolNode.input     = mapTensorNames(this, {nextOperatorInput}, TensorNameMap);
            maxPoolNode.attribute = [...
                makeAttributeProto('kernel_shape',  'INTS', poolSize),...
                makeAttributeProto('pads',          'INTS', paddingSize),...
                makeAttributeProto('strides',       'INTS', stride)
                ];
            % Check if there are 2 outputs
            if this.OpsetVersion >= 8 && ...
                isa(this.NNTLayer, 'nnet.cnn.layer.MaxPooling2DLayer') && ...
                this.NNTLayer.HasUnpoolingOutputs
            
                nextOperatorInput      	= [onnxName '_out'];
                indicesName             = [onnxName '_indices'];
                sizeName                = [onnxName '_size']; 
                
                % This creates a shape for the corresponding
                % MaxUnpool(X, indices, Shape(poolingInput))
                shapeNode = NodeProto; 
                shapeNode.op_type = 'Shape'; 
                shapeNode.name = sizeName;
                shapeNode.input = maxPoolNode.input;
                shapeNode.output = {sizeName};

                % Add renamed output tensors to name Map
                origOutName             = [this.NNTLayer.Name '/out'];
                origIndicesName         = [this.NNTLayer.Name '/indices'];
                origSizeName           = [this.NNTLayer.Name '/size']; 
                TensorNameMap(origIndicesName) = indicesName;
                TensorNameMap(origSizeName)   = sizeName;        
                
                % Set MaxPool node outputs
                maxPoolNode.output      = {nextOperatorInput, indicesName};
                hasSizeNode = true; 
            else 
                % No unpooling outputs
                nextOperatorInput           = onnxName;
                maxPoolNode.output          = {onnxName};
                origOutName                 = this.NNTLayer.Name;
                hasSizeNode = false; 
            end
            
            if hasSizeNode 
                nodeProto(end+1:end+2) = [shapeNode maxPoolNode];
            else
                nodeProto(end+1)       = maxPoolNode;
            end
            
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
            
            % Update layout map
            TensorNameMap(origOutName) = nextOperatorInput;
            TensorLayoutMap(nextOperatorInput) = outputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
