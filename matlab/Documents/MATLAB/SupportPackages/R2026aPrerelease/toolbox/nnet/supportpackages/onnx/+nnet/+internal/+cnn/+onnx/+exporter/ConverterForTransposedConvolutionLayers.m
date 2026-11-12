classdef ConverterForTransposedConvolutionLayers < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a transposedConv1dLayer, transposedConv2dLayer or transposedConv3dLayer into ONNX
        
    % Copyright 2018-2025 The MathWorks, Inc.

    methods
        function this = ConverterForTransposedConvolutionLayers(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
                     
            % The layer can be either a transposedConv1dLayer, transposedConv2dLayer or
            % transposedConv3dlayer. Determine what shape parameters to use        
            nntLayer                = this.NNTLayer;
            nntCrop                 = nntLayer.CroppingSize;

            [onnxName, ~] = legalizeNNTName(this, nntLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
            nextOperatorInput       = this.InputLayerNames{1};

            if isa(nntLayer, 'nnet.cnn.layer.TransposedConvolution1DLayer')
                if ismember(inputTensorLayout,{'snc','nch'})
                    dilations           = 1;
                    permWeightIdx       = [3 2 1];                    % DLT is HCF. ONNX is FCH.
                    cropping            = nntCrop([1 2]);             % nntCrop is [left right]  
                elseif ismember(inputTensorLayout,{'snch'})
                    dilations           = [1 1];
                    permWeightIdx       = [3 2 1 4];                   % DLT is HCFS. ONNX is FCHS.
                    cropping            = [nntCrop(1) 0 nntCrop(2) 0]; % nntCrop is [left right]  
                end
            elseif isa(nntLayer, 'nnet.cnn.layer.TransposedConvolution2DLayer')
                if ismember(inputTensorLayout,{'nchw'})              % No 'snch' for TransposedConvolution2DLayer
                dilations           = [1 1];
                permWeightIdx       = [4 3 1 2];
                cropping            = nntCrop([1 3 2 4]);           % nntCrop is [top bottom left right]          
                elseif ismember(inputTensorLayout,{'snchw'})
                    dilations           = [1 1 1];
                    permWeightIdx       = [4 3 1 2 5];
                    cropping            = nntCrop([1 3 0 2 4 0]);    % nntCrop is [top bottom left right]  
                end
            elseif isa(nntLayer, 'nnet.cnn.layer.TransposedConvolution3DLayer')
                if ismember(inputTensorLayout,{'nchwd'})              % No 'snchw' for TransposedConvolution3DLayer
                dilations           = [1 1 1];
                permWeightIdx       = [5 4 1 2 3];   
                cropping            = nntCrop([1 3 5 2 4 6]);       % nntCrop is [top left front; bottom right back]
                elseif ismember(inputTensorLayout,{'snchwd'})
                    dilations           = [1 1 1 1];
                    permWeightIdx       = [5 4 1 2 3 6];   
                    cropping            = nntCrop([1 3 5 0 2 4 6 0]); % nntCrop is [top left front; bottom right back]
                end
            else
                error(message("nnet_cnn_onnx:onnx:UnexpectedLayer", 'nnet.cnn.layer.TransposedConvolution1DLayer, nnet.cnn.layer.TransposedConvolution2DLayer', 'nnet.cnn.layer.TransposedConvolution3DLayer', class(this.NNTLayer)));
            end
            
            %  Permute Sequence input dimension to Spatial Dimension for
            %  Conv. Permute the Sequence dim from the first dim to the
            %  last dim
            [nodeProto, nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, 1);
            
            % Make the nodeProto
            newNode           = NodeProto;
            newNode.op_type	= 'ConvTranspose';
            newNode.name      = onnxName;
            input               = {...
                nextOperatorInput,... % X
                [onnxName '_W'],...
                [onnxName '_B']
                };
            newNode.input     = mapTensorNames(this, input(:)', TensorNameMap);
            newNode.output    = {onnxName};
            newNode.attribute = [...
                makeAttributeProto('group',        'INT',  numel(nntLayer.NumChannels)) ,...
                makeAttributeProto('dilations',    'INTS', dilations),...
                makeAttributeProto('kernel_shape', 'INTS', nntLayer.FilterSize),...
                makeAttributeProto('pads',         'INTS', cropping),...  
                makeAttributeProto('strides',      'INTS', nntLayer.Stride)
                ];
            
            % Make parameter Initializers for: W, B
            t1 = TensorProto;
            t1.name = [onnxName '_W'];     
            t1.data_type = TensorProto_DataType.FLOAT;
            permutedW = permute(nntLayer.Weights, permWeightIdx);	% NNT is HFC(1D) or HWFC(2D) or HWDFC(3D). ONNX is CFH(1D) or CFHW(2D) or CFHWD(3D).
            t1.raw_data = rawData(single(permutedW));
            t1.dims = dimVector(size(permutedW),numel(permWeightIdx));           
            
            t2 = TensorProto;
            t2.name = [onnxName '_B'];
            t2.data_type = TensorProto_DataType.FLOAT;
            t2.raw_data = rawData(single(squeeze(nntLayer.Bias)));
            t2.dims = dimVector(numel(nntLayer.Bias),1);            % NNT data: 1-1-numFilters
            
            parameterInitializers = [t1 t2];
              
            nodeProto(end+1)        = newNode;
            networkInputs           = [];
            networkOutputs          = [];
            
            % Update maps
            nextOperatorInput = onnxName;

            %Output a sequence if the input was a sequence by permuting
            %the sequence dimension from the last dim to the first dim
            [nodeProto, nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, 0);

            % Update maps
            TensorNameMap(this.NNTLayer.Name) = nextOperatorInput;
            TensorLayoutMap(nextOperatorInput)= inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
        
    end
end

