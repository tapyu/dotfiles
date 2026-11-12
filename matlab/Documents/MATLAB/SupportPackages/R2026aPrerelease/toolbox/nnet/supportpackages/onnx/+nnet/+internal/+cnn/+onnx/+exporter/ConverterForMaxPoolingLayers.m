classdef ConverterForMaxPoolingLayers < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a MaxPooling2dLayer or MaxPooling3dLayer into ONNX
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForMaxPoolingLayers(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
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
            %  MaxPool. 
            [nodeProto, nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, 1);

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
            
            maxPoolInput = mapTensorNames(this, {nextOperatorInput}, TensorNameMap);
            % Check if there are 2 outputs
            if this.OpsetVersion >= 8 && ...
                isa(this.NNTLayer, 'nnet.cnn.layer.MaxPooling2DLayer') && ...
                this.NNTLayer.HasUnpoolingOutputs
            
                nextOperatorInput      	= [onnxName '_out'];
                indicesName             = [onnxName '_indices'];
                sizeName                = [onnxName '_size']; 
                
                % This creates a shape for the corresponding
                % MaxUnpool(X, indices, Shape(poolingInput))
                shapeNode = createNodeProto(this, 'SISO', 'Shape', sizeName,...
                    maxPoolInput, {sizeName});

                % Add renamed output tensors to name Map
                origOutName             = [this.NNTLayer.Name '/out'];
                origIndicesName         = [this.NNTLayer.Name '/indices'];
                origSizeName           = [this.NNTLayer.Name '/size']; 
                TensorNameMap(origIndicesName) = indicesName;
                TensorNameMap(origSizeName)   = sizeName;        
                
                % Set MaxPool node outputs
                maxPoolOutput      = {nextOperatorInput, indicesName};
                hasSizeNode = true; 
            else 
                % No unpooling outputs
                nextOperatorInput           = onnxName;
                maxPoolOutput          = {onnxName};
                origOutName                 = this.NNTLayer.Name;
                hasSizeNode = false; 
            end
            % Create the maxpool node
            [maxPoolNode, ~] = createNodeProto(this, "MaxPool", onnxName,...
                maxPoolInput, maxPoolOutput, poolSize, paddingSize, stride);

            if hasSizeNode 
                nodeProto(end+1:end+2) = [shapeNode maxPoolNode];
            else
                nodeProto(end+1)       = maxPoolNode;
            end
            
            %Output a sequence if the input was a sequence by permuting
            %the sequence dimension from the last dim to the first dim
            [nodeProto, nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, 0);


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
