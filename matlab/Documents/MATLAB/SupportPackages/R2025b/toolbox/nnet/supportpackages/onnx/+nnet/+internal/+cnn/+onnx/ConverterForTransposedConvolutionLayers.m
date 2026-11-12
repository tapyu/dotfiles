classdef ConverterForTransposedConvolutionLayers < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a transposedConv2dLayer or transposedConv3dLayer into ONNX
        
    % Copyright 2018-2021 The MathWorks, Inc.

    methods
        function this = ConverterForTransposedConvolutionLayers(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
                     
            % The layer can be either a transposedConv2dLayer or
            % transposedConv3dlayer. Determine what shape parameters to use        
            nntLayer                = this.NNTLayer;
            nntCrop                 = nntLayer.CroppingSize;
            if isa(nntLayer, 'nnet.cnn.layer.TransposedConvolution2DLayer')
                dilations           = [1 1];
                permWeightIdx       = [4 3 1 2];
                cropping            = nntCrop([1 3 2 4]);           % nntCrop is [top bottom left right]          
            elseif isa(nntLayer, 'nnet.cnn.layer.TransposedConvolution3DLayer')
                dilations           = [1 1 1];
                permWeightIdx       = [5 4 1 2 3];   
                cropping            = nntCrop([1 3 5 2 4 6]);       % nntCrop is [top left front; bottom right back]
            else
                error(message("nnet_cnn_onnx:onnx:UnexpectedLayer", 'nnet.cnn.layer.TransposedConvolution2DLayer', 'nnet.cnn.layer.TransposedConvolution3DLayer', class(this.NNTLayer)));
            end
            
            [onnxName, nameChanged] = legalizeNNTName(this, nntLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
            
            % Make the nodeProto
            newNode           = NodeProto;
            newNode.op_type	= 'ConvTranspose';
            newNode.name      = onnxName;
            input               = {...
                this.InputLayerNames{1},... % X
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
            permutedW = permute(nntLayer.Weights, permWeightIdx);	% NNT is HWFC(2D) or HWDFC(3D). ONNX is CFHW(2D) or CFHWD(3D).
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
            outputTensorName = onnxName;
            TensorNameMap(nntLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
        
    end
end

