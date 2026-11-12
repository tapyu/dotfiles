classdef ConverterForConvolutionLayers < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a convolution2dLayer or convolution3dLayer into ONNX
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForConvolutionLayers(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            % The layer can be either a convolution1dlayer, convolution2dLayer or
            % convolution3dlayer. Determine how to permute the weights and
            % padding size, and what the shape of the tensor will be          
            nntLayer            = this.NNTLayer;
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            nextOperatorInput   = this.InputLayerNames{1};

            % Exporting convolutional layers operating over SSSCBT data is
            % currently unsupported
            if isequal(inputTensorLayout,'snchwd')
                error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end

            [permWeightIdx, permutedPadding, filterSize,...
                dilationFactor, stride, nDims] = getPermutationAndAttributes(this,inputTensorLayout);

            %  Permute Sequence input dimension to Spatial Dimension for
            %  Conv. Permute the Sequence dim from the first dim to the
            %  last dim
            [nodeProto, nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, 1);
            
            inits     = [];
            paddingMode = this.NNTLayer.PaddingMode;
            paddingValue = this.NNTLayer.PaddingValue;

            if isequal(paddingValue,"symmetric-include-edge")
                error(message("nnet_cnn_onnx:onnx:ConvPaddingValueUnsupportedForExport", class(this.NNTLayer)));
            elseif isequal(paddingValue,"symmetric-exclude-edge")
                padOnnxMode = 'reflect';
            elseif isequal(paddingValue,"replicate")
                padOnnxMode = 'edge';
            else
                padOnnxMode    = 'constant';
            end

            [onnxName, ~]   = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName        = makeUniqueName({nodeProto.name}, onnxName);
            padInput        = mapTensorNames(this, {nextOperatorInput}, TensorNameMap);
            padOnnxName = [onnxName,'_Pad'];

            if isequal(paddingMode, 'same') && this.OpsetVersion > 10
                [nodeProto, inits] = createOperatorsForSamePaddingMode(this, nodeProto, inits, filterSize,...
                dilationFactor, stride, nDims, padOnnxName, padInput, onnxName, padOnnxMode, paddingValue);
                % update conv input: padOnnxName is the input and padding
                % has size zero.
                nextOperatorInput = padOnnxName;
                permutedPadding   = 0*permutedPadding;
            elseif paddingValue~=0
                % The ONNX pad operator requires as many elements as twice
                % the number of dimensions. add zero-sized padding
                % corresponding to the batch and channel dimensions.
                beginPadding = [zeros(1,2) permutedPadding(1:nDims)];
                endPadding   = [zeros(1,2) permutedPadding(nDims+1:end)];
                pads         = [beginPadding endPadding];

                [padNode, padInits] = createNodeProto(this, 'Pad', padOnnxName,...
                    padInput, {padOnnxName}, pads, padOnnxMode, paddingValue);                
                nodeProto(end+1) = padNode;
                inits = [inits, padInits];
                % update conv input: padOnnxName is the input and padding
                % has size zero.
                nextOperatorInput = padOnnxName;
                permutedPadding   = 0*permutedPadding;
            end

            % Make the nodeProto
            WName                   = [onnxName '_W'];
            BName                   = [onnxName '_B'];
            convNode                = NodeProto;
            convNode.op_type        = 'Conv';
            convNode.name           = onnxName;
            convNode.input          = mapTensorNames(this, {nextOperatorInput, WName, BName}, TensorNameMap);
            convNode.output         = {onnxName};
            convNode.attribute = [...
                    makeAttributeProto('group',        'INT',  numel(nntLayer.NumChannels)),... % See alexnet for an example where this is >1
                    makeAttributeProto('dilations',    'INTS', dilationFactor),...
                    makeAttributeProto('kernel_shape', 'INTS', filterSize),...
                    makeAttributeProto('pads',         'INTS', permutedPadding),...
                    makeAttributeProto('strides',      'INTS', stride)
                    ];
            % Make parameter Initializers for: W, B
            t1              = TensorProto;
            t1.name         = WName;
            t1.data_type	= TensorProto_DataType.FLOAT;
            permutedW       = permute(this.NNTLayer.Weights, permWeightIdx);
            t1.raw_data     = rawData(single(permutedW));
            t1.dims         = dimVector(size(permutedW),numel(permWeightIdx));
            
            t2              = TensorProto;
            t2.name         = BName;
            t2.data_type    = TensorProto_DataType.FLOAT;
            t2.raw_data     = rawData(single(squeeze(this.NNTLayer.Bias)));
            t2.dims         = dimVector(numel(this.NNTLayer.Bias),1);            % NNT data: numFilters-1 (1D)  or 1-1-numFilters(2D) or 1-1-1-numFilters(3D)
            

            nodeProto(end+1)        = convNode;
            parameterInitializers   = [inits t1 t2];
            networkInputs           = [];
            networkOutputs          = [];
            
            nextOperatorInput       = onnxName;

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

    methods (Access = private)
        function [permWeightIdx, permutedPadding, filterSize, dilationFactor,...
                stride, nDims] =  getPermutationAndAttributes(this, inputTensorLayout)
             nntLayer = this.NNTLayer;
             if isa(nntLayer, 'nnet.cnn.layer.Convolution1DLayer')
                if ismember(inputTensorLayout,{'snc','nch'})
                    %Permutations
                    %Weights
                    permWeightIdx       = [3 2 1]; % DLT is HCF. ONNX is FCH
                    %Padding
                    permPadIdx          = [1 2]; % Pad is [l r], we want [l r]
                    permutedPadding     = nntLayer.PaddingSize(permPadIdx);
                    
                    %ConvAttributes
                    filterSize          = nntLayer.FilterSize;
                    dilationFactor      = nntLayer.DilationFactor;
                    stride              = nntLayer.Stride;
                    nDims               = 1;
                elseif ismember(inputTensorLayout,{'snch'})
                    %Permutations
                    permWeightIdx       = [3 2 1 4]; % DLT is HCFS. ONNX is FCHS.
                    permutedPadding     = [nntLayer.PaddingSize(1) 0 nntLayer.PaddingSize(2) 0];
                    
                    %Attributes
                    filterSize          = [nntLayer.FilterSize 1];
                    dilationFactor      = [nntLayer.DilationFactor 1];
                    stride              = [nntLayer.Stride 1];
                    nDims               = 2;
                end
             elseif isa(nntLayer, 'nnet.cnn.layer.Convolution2DLayer')
                 if ismember(inputTensorLayout,{'nchw','snch'})
                    %Permutations
                    permWeightIdx       = [4 3 1 2]; % DLT is HWCF. ONNX is FCHW.
                    permPadIdx          = [1 3 2 4]; % Pad is [t l; b r], we want [t l b r]
                    permutedPadding     = nntLayer.PaddingSize(permPadIdx);

                    %Attributes
                    filterSize          = nntLayer.FilterSize;
                    dilationFactor      = nntLayer.DilationFactor;
                    stride              = nntLayer.Stride;
                    nDims = 2;
                 elseif ismember(inputTensorLayout,{'snchw'})
                    %Permutations
                    permWeightIdx       = [4 3 1 2 5]; % DLT is HWCFS. ONNX is FCHWS.
                    permutedPadding     = [nntLayer.PaddingSize(1) nntLayer.PaddingSize(3) 0 nntLayer.PaddingSize(2) nntLayer.PaddingSize(4) 0];

                    %Attributes
                    filterSize          = [nntLayer.FilterSize 1];
                    dilationFactor      = [nntLayer.DilationFactor 1];
                    stride              = [nntLayer.Stride 1];
                    nDims = 3;
                 end
             elseif isa(nntLayer, 'nnet.cnn.layer.Convolution3DLayer')
                %Permutations
                permWeightIdx       = [5 4 1 2 3]; % DLT is HWDCF. ONNX is FCHWD.
                permPadIdx          = [1 3 5 2 4 6]; % Pad is [t l f; b r k], we want [t l f b r k]
                permutedPadding     = nntLayer.PaddingSize(permPadIdx);     
                
                %Attributes
                filterSize          = nntLayer.FilterSize;
                dilationFactor      = nntLayer.DilationFactor;
                stride              = nntLayer.Stride;
                nDims = 3;
             else
                    error(message("nnet_cnn_onnx:onnx:UnexpectedLayer", 'nnet.cnn.layer.Convolution2DLayer', 'nnet.cnn.layer.Convolution3DLayer', class(this.NNTLayer)));
             end
        end
        function [nodeProto, inits] = createOperatorsForSamePaddingMode(this, nodeProto, inits, filterSize,...
                dilationFactor, stride, nDims, padOnnxName, padInput, onnxName, padOnnxMode, paddingValue)
            import nnet.internal.cnn.onnx.*
            % dilatedFilterSize = (FilterSize - 1) .* DilationFactor + 1;
            % desiredOutputSize = ceil(inputSize./stride);
            % desiredPaddedInputSize = (desiredOutputSize - 1) .* stride + dilatedFilterSize;
            % totalPaddingNeeded = desiredPaddedInputSize - inputSize;
            % % Need to ensure padding is positive.
            % totalPaddingNeeded(totalPaddingNeeded < 0) = 0;
            dilatedFilterSize = (filterSize - 1) .* dilationFactor + 1;
            % size input
            inputFeatureShapeIntName = [onnxName '_InputShapeInt'];
            [inputFeatureShapeIntNode, ~] = createNodeProto(this, 'SISO', 'Shape',...
                inputFeatureShapeIntName, padInput, {inputFeatureShapeIntName});
            % cast the size output to be non-int type for division
            inputFeatureShapeName = [onnxName '_InputShape'];
            [inputFeatureShapeNode, ~] = createNodeProto(this, 'Cast', inputFeatureShapeName, ...
            {inputFeatureShapeIntName}, {inputFeatureShapeName},1);

            % create stride constant
            StrideName = [onnxName '_Stride'];
            StrideInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(StrideName, ...
            [2+size(stride,2)], [1,1,stride], TensorProto_DataType.FLOAT);
            % div (input,stride)
            DivName = [onnxName '_Div'];
            [DivNode, ~] = createNodeProto(this, 'SISO', 'Div', DivName, ...
            {inputFeatureShapeName,StrideName}, {DivName});
            % ceil (div_ouput)
            CeilName = [onnxName '_Ceil'];
            [CeilNode, ~] = createNodeProto(this, 'SISO', 'Ceil', CeilName, ...
            {DivName}, {CeilName});
            % create 1 constant
            OneName = [onnxName '_One'];
            OneInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(OneName, ...
            [1], 1, TensorProto_DataType.FLOAT);
            % minus(ceil_out,1)
            SubName = [onnxName '_Sub'];
            [SubNode, ~] = createNodeProto(this, 'SISO', 'Sub', SubName, ...
            {CeilName,OneName}, {SubName});
            % mul(minus_out,stride)
            MulName = [onnxName '_Mul'];
            [MulNode, ~] = createNodeProto(this, 'SISO', 'Mul', MulName, ...
            {SubName,StrideName}, {MulName});
            % create filter or pool constant
            FilterName = [onnxName '_Filter'];
            FilterInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(FilterName, ...
            [2+size(dilatedFilterSize,2)], [1,1,dilatedFilterSize], TensorProto_DataType.FLOAT);
            % add(mul_out,filter)
            AddName = [onnxName '_Add'];
            [AddNode, ~] = createNodeProto(this, 'SISO', 'Add', AddName, ...
            {MulName,FilterName}, {AddName});
            % minus(add_out,size_out)
            Sub1Name = [onnxName '_Sub1'];
            [Sub1Node, ~] = createNodeProto(this, 'SISO', 'Sub', Sub1Name, ...
            {AddName,inputFeatureShapeName}, {Sub1Name});
            % relu(minus_out)
            ReluIntName = [onnxName '_ReluInt'];
            [ReluIntNode, ~] = createNodeProto(this, 'SISO', 'Relu', ReluIntName, ...
            {Sub1Name}, {ReluIntName});
            % cast the relu output to be non-int type for division
            ReluName = [onnxName '_Relu'];
            [ReluNode, ~] = createNodeProto(this, 'Cast', ReluName, ...
            {ReluIntName}, {ReluName},1);

            % create 0 constant for future use
            ZeroName = [onnxName '_Zero'];
            ZeroInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(ZeroName, ...
            [1], 0, TensorProto_DataType.FLOAT);

            %create names for future use
            paddingSizeName = [onnxName '_paddingSize'];
            splitNodeName = [onnxName '_Split'];
            BCPaddingName = [onnxName '_BCPadding'];
            totalPaddingNeeded1Name = [onnxName '_totalPaddingNeeded1'];
            totalPaddingNeeded2Name = [onnxName '_totalPaddingNeeded2'];
            totalPaddingNeeded3Name = [onnxName '_totalPaddingNeeded3'];

            inits = [inits,StrideInit,OneInit,FilterInit,ZeroInit];
            nodeProto = [nodeProto,inputFeatureShapeIntNode,inputFeatureShapeNode,...
                DivNode,CeilNode,SubNode,MulNode,AddNode,Sub1Node,ReluIntNode,ReluNode];

            % Construct the padding vector based on the dimensionality of the input
            % topPadding = floor(totalPaddingNeeded(1)/2);
            % bottomPadding = ceil(totalPaddingNeeded(1)/2);
            % leftPadding = floor(totalPaddingNeeded(2)/2);
            % rightPadding = ceil(totalPaddingNeeded(2)/2);
            % frontPadding = floor(totalPaddingNeeded(3)/2);
            % backPadding = ceil(totalPaddingNeeded(3)/2);   
            % beginPadding = [zeros(1,2) permutedPadding(1:nDims)];
            % endPadding   = [zeros(1,2) permutedPadding(nDims+1:end)];
            % pads         = [beginPadding endPadding];

            % split the total padding needed
            if isequal(nDims,1)
                % conv1d
                [splitNode, splitInit] = createNodeProto(this, 'Split', splitNodeName, ...
                {ReluName}, {BCPaddingName,totalPaddingNeeded1Name},0,[2,1]);
                [topBotNodes,topBotInit,topPaddingName, bottomPaddingName] =...
                    createPadSizeNodes(this,totalPaddingNeeded1Name);
                [paddingSizeNode, ~] = createNodeProto(this, 'Concat', paddingSizeName,...
                    {ZeroName,ZeroName,topPaddingName,...
                    ZeroName,ZeroName,bottomPaddingName}, {paddingSizeName}, 0);
                inits = [inits splitInit topBotInit];
                nodeProto = [nodeProto, splitNode, topBotNodes, paddingSizeNode];
            elseif isequal(nDims,2)
                % conv2d
                [splitNode, splitInit] = createNodeProto(this, 'Split', splitNodeName, ...
                {ReluName}, {BCPaddingName,totalPaddingNeeded1Name,totalPaddingNeeded2Name},0,[2,1,1]);
                [topBotNodes,topBotInit,topPaddingName, bottomPaddingName] =...
                    createPadSizeNodes(this,totalPaddingNeeded1Name);
                [leftRightNodes,leftRightInits,leftPaddingName, rightPaddingName] =...
                    createPadSizeNodes(this,totalPaddingNeeded2Name);
                [paddingSizeNode, ~] = createNodeProto(this, 'Concat', paddingSizeName,...
                    {ZeroName,ZeroName,topPaddingName,leftPaddingName,...
                    ZeroName,ZeroName,bottomPaddingName,rightPaddingName}, {paddingSizeName}, 0);
                inits = [inits,splitInit, topBotInit, leftRightInits];
                nodeProto = [nodeProto, splitNode, topBotNodes, ...
                    leftRightNodes, paddingSizeNode];
            else
                % conv3d
                [splitNode, splitInit] = createNodeProto(this, 'Split', splitNodeName, ...
                {ReluName}, {BCPaddingName,totalPaddingNeeded1Name,...
                totalPaddingNeeded2Name, totalPaddingNeeded3Name},0,[2,1,1,1]);
                [topBotNodes,topBotInit,topPaddingName, bottomPaddingName] =...
                    createPadSizeNodes(this,totalPaddingNeeded1Name);
                [leftRightNodes,leftRightInits,leftPaddingName, rightPaddingName] =...
                    createPadSizeNodes(this,totalPaddingNeeded2Name);
                [frontBackNodes,frontBackInits,frontPaddingName, backPaddingName] =...
                    createPadSizeNodes(this,totalPaddingNeeded3Name);
                [paddingSizeNode, ~] = createNodeProto(this, 'Concat',paddingSizeName,...
                    {ZeroName,ZeroName,topPaddingName,leftPaddingName,frontPaddingName,...
                    ZeroName,ZeroName,bottomPaddingName,rightPaddingName,...
                    backPaddingName}, {paddingSizeName}, 0);
                inits = [inits, splitInit, topBotInit, ...
                    leftRightInits,frontBackInits];
                nodeProto = [nodeProto, splitNode, topBotNodes, ...
                    leftRightNodes, frontBackNodes, paddingSizeNode];
            end
            paddingSizeIntName = [onnxName '_paddingSizeInt'];
            [paddingSizeIntNode, ~] = createNodeProto(this, 'Cast', paddingSizeIntName, ...
            {paddingSizeName}, {paddingSizeIntName},7);
            [padNode, padInits] = createNodeProto(this, 'Pad', padOnnxName,...
                padInput, {padOnnxName}, paddingSizeIntName, padOnnxMode, paddingValue);    
            nodeProto = [nodeProto, paddingSizeIntNode, padNode];
            inits = [inits, padInits];
        end
        function [nodes, inits, firstNodeName, secondNodeName] = createPadSizeNodes(this,totalPaddingNeededName)
            import nnet.internal.cnn.onnx.*
            % create a constant 2 node
            TwoName = [totalPaddingNeededName '_Two'];
            TwoInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(TwoName, ...
            [1], 2, TensorProto_DataType.FLOAT);
            % div
            DivName = [totalPaddingNeededName '_Div'];
                [DivNode, ~] = createNodeProto(this, 'SISO','Div', DivName, ...
                {totalPaddingNeededName,TwoName}, {DivName});
            % floor
            firstNodeName = [totalPaddingNeededName '_Floor'];
                [firstNode, ~] = createNodeProto(this, 'SISO','Floor', firstNodeName, ...
                {DivName}, {firstNodeName});
            % ceil
            secondNodeName = [totalPaddingNeededName '_Ceil'];
                [secondNode, ~] = createNodeProto(this, 'SISO', 'Ceil', secondNodeName, ...
                {DivName}, {secondNodeName});
            nodes = [DivNode,firstNode,secondNode];
            inits = [TwoInit];
        end
    end
end

