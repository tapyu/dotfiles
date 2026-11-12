classdef NNTLayerConverter
    % Class to convert a NNT Layer to a nnet.internal.cnn.onnx.NodeProto
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    %% API
    properties(Dependent)
        NNTLayer            % The NNT layer being converted.
        InputLayerNames     % (cellstr) The names of the layers (really tensors) sending input connections to this layer.
        InputLayerSizes     % (cell array) Sizes of the layers (really tensors) sending input connections to this layer.
        FullInputSizes      % (cell array) Sizes of the inputs to a layer, including all S,C,B, or T dimensions.        
        OutputSize          % Size of the output tensor of this layer.
    end
    
    properties
        OpsetVersion        % Integer
        IsDanglingLayer     % Always false for any layer in a DAGNetwork since DAGnetwork must have outputlayer(s)
                            % True for Layergraph if is leaf layer and current layergraph does not have output layer 
                            % True for dlnetwork if is leaf layer since does not have output
        IsRecurrentNetwork
        BatchSizeToExport   % This can be a number or a string e.g., "BatchSize"
    end
    
    %% Override 'toOnnx()' to convert an NNT layer to ONNX.
    methods
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % Inputs:
        %   * nodeProto is an array of nnet.internal.cnn.onnx.NodeProto.
            %   * TensorNameMap is a containers.Map that translates between NNT
            %   output names and ONNX tensor names.
            %   * TensorLayoutMap is a containers.Map that translates between
            %   ONNX tensor names and ONNX tensor layout strings such as
            %   'nchw', 'snc, etc.
            %
            % Outputs:
            %   * nodeProto in the output list is the updated array with additions from this layer.
        %   * parameterInitializers is an array of  nnet.internal.cnn.onnx.TensorProto.
        %   * networkInputs is an array of nnet.internal.cnn.onnx.ValueInfoProto,
        %   which should be present only when this layer defines input to the network as a whole.
        %   * networkOutputs is an array of nnet.internal.cnn.onnx.ValueInfoProto,
        %   which should be present only when this layer defines output of the network as a whole.
            %   * TensorNameMap in the output list is the updated map with additions from this layer.
            %   * TensorLayoutMap in the output list is the updated map with additions from this layer.
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];
        end
    end
    
    methods(Static)
        function converter = makeLayerConverter(layerAnalyzer, OpsetVersion, IsRecurrentNetwork, IsDanglingLayer, BatchSizeToExport)
            % Factory method to create a converter for a specific layer.
            import nnet.internal.cnn.onnx.exporter.*
            assert(isa(layerAnalyzer, 'nnet.internal.cnn.analyzer.util.LayerAnalyzer'));
            nntLayer = layerAnalyzer.ExternalLayer;
            assert(isa(nntLayer, 'nnet.cnn.layer.Layer'));
            
            % DLT Layers
            if isa(nntLayer, 'nnet.cnn.layer.AdaptiveAveragePooling2DLayer')
                converterConstructor = @ConverterForAdaptiveAveragePooling2dLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.AdditionLayer')
                converterConstructor = @ConverterForAdditionLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.AveragePooling1DLayer')
                converterConstructor = @ConverterForAveragePoolingLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.AveragePooling2DLayer')
                converterConstructor = @ConverterForAveragePoolingLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.AveragePooling3DLayer')
                converterConstructor = @ConverterForAveragePoolingLayers;                
            elseif isa(nntLayer, 'nnet.cnn.layer.BatchNormalizationLayer')
                converterConstructor = @ConverterForBatchNormalizationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.BiLSTMLayer')
                converterConstructor = @ConverterForBiLSTMLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ClassificationOutputLayer')
                converterConstructor = @ConverterForClassificationOutputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ClippedReLULayer')
                converterConstructor = @ConverterForClippedReLULayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ComplexFullyConnectedLayer')
                converterConstructor = @ConverterForComplexFullyConnectedLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ComplexReLULayer')
                converterConstructor = @ConverterForComplexReLULayer;    
            elseif isa(nntLayer, 'nnet.cnn.layer.ConcatenationLayer')
                converterConstructor = @ConverterForConcatenationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.Convolution1DLayer')
                converterConstructor = @ConverterForConvolutionLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.Convolution2DLayer')
                converterConstructor = @ConverterForConvolutionLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.Convolution3DLayer')
                converterConstructor = @ConverterForConvolutionLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.CrossChannelNormalizationLayer')
                converterConstructor = @ConverterForCrossChannelNormalizationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.DepthConcatenationLayer')
                converterConstructor = @ConverterForDepthConcatenationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.DropoutLayer')
                converterConstructor = @ConverterForDropoutLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ELULayer')
                converterConstructor = @ConverterForEluLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.FeatureInputLayer')
                converterConstructor = @ConverterForFeatureInputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.FlattenLayer')
                converterConstructor = @ConverterForFlattenLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.FullyConnectedLayer')
                converterConstructor = @ConverterForFullyConnectedLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.GlobalAveragePooling1DLayer')
                converterConstructor = @ConverterForGlobalAveragePooling1dLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.GlobalAveragePooling2DLayer')
                converterConstructor = @ConverterForGlobalAveragePooling2dLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.GlobalAveragePooling3DLayer')
                converterConstructor = @ConverterForGlobalAveragePooling3dLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.GlobalMaxPooling1DLayer')
                converterConstructor = @ConverterForGlobalMaxPooling1dLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.GlobalMaxPooling2DLayer')
                converterConstructor = @ConverterForGlobalMaxPooling2dLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.GlobalMaxPooling3DLayer')
                converterConstructor = @ConverterForGlobalMaxPooling3dLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.GroupedConvolution2DLayer')
                converterConstructor = @ConverterForGroupedConvolution2DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.GroupNormalizationLayer')
                if nntLayer.NumChannels == nntLayer.NumGroups
                    converterConstructor = @ConverterForGroupNormalizationLayer;
                else
                    error(message('nnet_cnn_onnx:onnx:InvalidNumGroups', nntLayer.Name));
                end
            elseif isa(nntLayer, 'nnet.cnn.layer.InputLayer')
                converterConstructor = @ConverterForInputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.InstanceNormalizationLayer')
                converterConstructor = @ConverterForInstanceNormalizationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.GRULayer')
                converterConstructor = @ConverterForGRULayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.GRUProjectedLayer')
                converterConstructor = @ConverterForGRUProjectedLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.IdentityLayer')
                converterConstructor = @ConverterForIdentityLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ImageInputLayer')
                converterConstructor = @ConverterForImageInputLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.Image3DInputLayer')
                converterConstructor = @ConverterForImageInputLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.LayerNormalizationLayer')
                converterConstructor = @ConverterForLayerNormalizationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.LeakyReLULayer')
                converterConstructor = @ConverterForLeakyReLULayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.LSTMLayer')
                converterConstructor = @ConverterForLSTMLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.LSTMProjectedLayer')
                converterConstructor = @ConverterForLSTMProjectedLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.MaxPooling1DLayer')
                converterConstructor = @ConverterForMaxPoolingLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.MaxPooling2DLayer')
                converterConstructor = @ConverterForMaxPoolingLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.MaxPooling3DLayer')
                converterConstructor = @ConverterForMaxPoolingLayers;                
            elseif isa(nntLayer, 'nnet.cnn.layer.MaxUnpooling2DLayer')
                converterConstructor = @ConverterForMaxUnpooling2DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.RealToComplexLayer')
                converterConstructor = @ConverterForRealToComplexLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.MultiplicationLayer')
                converterConstructor = @ConverterForMultiplicationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.PointCloudInputLayer')
                converterConstructor = @ConverterForPointCloudInputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.PReLULayer')
                converterConstructor = @ConverterForPReLULayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.RegressionOutputLayer')
                converterConstructor = @ConverterForRegressionOutputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ReLULayer')
                converterConstructor = @ConverterForReLULayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SequenceFoldingLayer')
                converterConstructor = @ConverterForSequenceFoldingLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SequenceInputLayer')
                converterConstructor = @ConverterForSequenceInputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SequenceUnfoldingLayer')
                converterConstructor = @ConverterForSequenceUnfoldingLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ScalingLayer')
                converterConstructor = @ConverterForScalingLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SigmoidLayer')
                converterConstructor = @ConverterForSigmoidLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SoftmaxLayer')
                converterConstructor = @ConverterForSoftmaxLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SoftplusLayer')
                converterConstructor = @ConverterForSoftplusLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SpatialDropoutLayer')
                converterConstructor = @ConverterForSpatialDropoutLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SpectralConvolution1DLayer')
                converterConstructor = @ConverterForSpectralConvolution1DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SpectralConvolution2DLayer')
                converterConstructor = @ConverterForSpectralConvolution2DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SpectralConvolution3DLayer')
                converterConstructor = @ConverterForSpectralConvolution3DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ComplexToRealLayer')
                converterConstructor = @ConverterForComplexToRealLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SwishLayer')
                converterConstructor = @ConverterForSwishLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.TransposedConvolution1DLayer')
                converterConstructor = @ConverterForTransposedConvolutionLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.TransposedConvolution2DLayer')
                converterConstructor = @ConverterForTransposedConvolutionLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.TransposedConvolution3DLayer')
                converterConstructor = @ConverterForTransposedConvolutionLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.TanhLayer')
                converterConstructor = @ConverterForTanhLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.ZReLULayer')
                converterConstructor = @ConverterForZReLULayer;
                
                % CVST Layers
            elseif isa(nntLayer, 'nnet.cnn.layer.Crop2DLayer')
                converterConstructor = @ConverterForCrop2DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.PixelClassificationLayer')
                converterConstructor = @ConverterForPixelClassificationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.RCNNBoxRegressionLayer')
                converterConstructor = @ConverterForRCNNBoxRegressionLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ROIInputLayer')
                converterConstructor = @ConverterForROIInputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ROIMaxPooling2DLayer')
                converterConstructor = @ConverterForROIMaxPooling2DLayer;
           
                % IPT Layers
            elseif isa(nntLayer, 'nnet.cnn.layer.DepthToSpace2DLayer')
                converterConstructor = @ConverterForDepthToSpace2DLayer;                
            elseif isa(nntLayer, 'nnet.cnn.layer.Resize2DLayer')
                converterConstructor = @ConverterForResizeLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.Resize3DLayer')
                converterConstructor = @ConverterForResizeLayers;
            elseif isa(nntLayer, 'nnet.cnn.layer.SpaceToDepthLayer')
                converterConstructor = @ConverterForSpaceToDepthLayer;                 
                
                % Keras importer layers
            elseif isa(nntLayer, 'nnet.keras.layer.ClipLayer')
                converterConstructor = @ConverterForKerasClipLayer;
            elseif isa(nntLayer, 'nnet.keras.layer.FlattenCStyleLayer')
                converterConstructor = @ConverterForKerasFlattenCStyleLayer;
            elseif isa(nntLayer, 'nnet.keras.layer.GlobalAveragePooling2dLayer')
                converterConstructor = @ConverterForKerasGlobalAveragePooling2dLayer;
            elseif isa(nntLayer, 'nnet.keras.layer.SigmoidLayer')
                converterConstructor = @ConverterForKerasSigmoidLayer;
            elseif isa(nntLayer, 'nnet.keras.layer.TanhLayer')
                converterConstructor = @ConverterForTanhLayers;
            elseif isa(nntLayer, 'nnet.keras.layer.ZeroPadding1dLayer')
                converterConstructor = @ConverterForKerasZeroPadding1dLayer;
            elseif isa(nntLayer, 'nnet.keras.layer.ZeroPadding2dLayer')
                converterConstructor = @ConverterForKerasZeroPadding2dLayer;
                
                % ONNX converter layers
            elseif isa(nntLayer, 'nnet.onnx.layer.CustomOutputLayer')
                converterConstructor = @ConverterForONNXCustomOutputLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.ClipLayer')
                converterConstructor = @ConverterForONNXClipLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.FlattenLayer')
                converterConstructor = @ConverterForONNXFlattenLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.FlattenInto2dLayer')
                converterConstructor = @ConverterForONNXFlattenInto2dLayer;                
            elseif isa(nntLayer, 'nnet.onnx.layer.GlobalAveragePooling2dLayer')
                converterConstructor = @ConverterForGlobalAveragePooling2dLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.GlobalMaxPooling2dLayer')
                converterConstructor = @ConverterForGlobalMaxPooling2dLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.IdentityLayer')
                converterConstructor = @ConverterForONNXIdentityLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.ImageScalerLayer')
                converterConstructor = @ConverterForONNXImageScalerLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.ElementwiseAffineLayer')
                converterConstructor = @ConverterForONNXElementwiseAffineLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.PreluLayer')
                converterConstructor = @ConverterForONNXPreluLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.ScalingFactorLayer')
                converterConstructor = @ConverterForONNXScalingFactorLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.SigmoidLayer')
                converterConstructor = @ConverterForONNXSigmoidLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.TanhLayer')
                converterConstructor = @ConverterForTanhLayers;
                
                % Densenet201 layers
            elseif isa(nntLayer, 'nnet.densenet201.layer.DenseNet201PreprocessingLayer')
                converterConstructor = @ConverterForDenseNet201PreprocessingLayer;
                
                % Inceptionresnetv2 layers
            elseif isa(nntLayer, 'nnet.inceptionresnetv2.layer.ScalingLayer')
                converterConstructor = @ConverterForInceptionResNetV2ScalingLayer;
            elseif isa(nntLayer, 'nnet.inceptionresnetv2.layer.ScalingFactorLayer')
                converterConstructor = @ConverterForInceptionResNetV2ScalingFactorLayer;
                
                % Inceptionv3 layers
            elseif isa(nntLayer, 'nnet.inceptionv3.layer.ScalingLayer')
                converterConstructor = @ConverterForInceptionV3ScalingLayer;
                
                % mobilenetv2 layers
            elseif isa(nntLayer, 'nnet.mobilenetv2.layer.MobileNetV2PreprocessingLayer')
                converterConstructor = @ConverterForMobileNetV2PreprocessingLayer;
                
                % nasnetmobile layers
            elseif isa(nntLayer, 'nnet.nasnetmobile.layer.NASNetMobileCrop2dLayer')
                converterConstructor = @ConverterForNASNetMobileCrop2dLayer;
            elseif isa(nntLayer, 'nnet.nasnetmobile.layer.NASNetMobilePreprocessingLayer')
                converterConstructor = @ConverterForNASNetMobilePreprocessingLayer;
            elseif isa(nntLayer, 'nnet.nasnetmobile.layer.NASNetMobileZeroPadding2dLayer')
                converterConstructor = @ConverterForNASNetMobileZeroPadding2dLayer;
                
                % Resnet18 layers
            elseif isa(nntLayer, 'nnet.resnet18.layer.ResNet18PreprocessingLayer')
                converterConstructor = @ConverterForResNet18PreprocessingLayer;
                
                % xception layers
            elseif isa(nntLayer, 'nnet.xception.layer.XceptionPreprocessingLayer')
                converterConstructor = @ConverterForXceptionPreprocessingLayer;
                
                % YOLO v2 layers
            elseif isa(nntLayer, 'nnet.cnn.layer.YOLOv2ReorgLayer')
                converterConstructor = @ConverterForYOLOv2ReorgLayer;                  
            elseif isa(nntLayer, 'nnet.cnn.layer.YOLOv2TransformLayer')
                converterConstructor = @ConverterForYOLOv2TransformLayer;       
            elseif isa(nntLayer, 'nnet.cnn.layer.YOLOv2OutputLayer')
                converterConstructor = @ConverterForYOLOv2OutputLayer;                    
                
            % Unknown (user-defined) custom layers
            %             elseif layerAnalyzer.IsCustomLayer
            %                 converterConstructor = @ConverterForCustomLayer;
            %                 warning(message('nnet_cnn_onnx:onnx:ExportingCustomLayer', class(nntLayer)));
            
            % Unsupported layers:
            elseif layerAnalyzer.IsOutputLayer
                converterConstructor = @ConverterForUnsupportedOutputLayer;
            elseif layerAnalyzer.IsInputLayer
                error(message('nnet_cnn_onnx:onnx:ExportUnsupportedInputLayer', class(layerAnalyzer.ExternalLayer)));
            else
                converterConstructor = @ConverterForUnsupportedLayer;
                nnet.internal.cnn.onnx.util.iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:ExportingPlaceholderLayer', class(nntLayer));
            end
            
            converter = converterConstructor(layerAnalyzer);
            converter.OpsetVersion = OpsetVersion;
            converter.IsRecurrentNetwork = IsRecurrentNetwork;
            converter.IsDanglingLayer = IsDanglingLayer;
            converter.BatchSizeToExport = BatchSizeToExport;
        end
    end
    
    %% INTERNALS
    properties(Access=private)
        LayerAnalyzer       % A LayerAnalyzer for the NNT layer being converted.
    end
    
    methods
        function this = NNTLayerConverter(layerAnalyzer)
            this.LayerAnalyzer = layerAnalyzer;
        end
        
        function [name, changed] = legalizeNNTName(~, nntName)
            % Make the NNT name a legal ONNX (C) name. Replace all chars
            % that are not letters or digits with underscores. If it starts
            % with a digit, prepend an underscore.
            name = nntName;
            toReplace = ~(isletter(name) | isdigit(name));
            name(toReplace) = '_';
            if isdigit(name(1))
                name = ['_' name];
            end
            changed = ~isequal(name, nntName);
        end
        
        function namesCell = mapTensorNames(~, namesCell, TensorNameMap)
            assert(iscell(namesCell));
            for i = 1:numel(namesCell)
                if isKey(TensorNameMap, namesCell{i})
                    namesCell{i} = TensorNameMap(namesCell{i});
                end
            end
        end
        
        function layer = get.NNTLayer(this)
            layer = this.LayerAnalyzer.ExternalLayer;
        end
        
        function inputNames = get.InputLayerNames(this)
            inputNames = this.LayerAnalyzer.Inputs.Source;
            inputNames = cellfun(@char, inputNames, 'UniformOutput', false);
        end
        
        function inputSizes = get.InputLayerSizes(this)
            % This method will only return the sizes of S and C
            % dimensions, and trailing singleton sizes may be dropped.
            inputSizes = this.LayerAnalyzer.Inputs.Size;
        end

        function inputSizes = get.FullInputSizes(this)
            % This method returns the sizes of all dlarray labels. Trailing
            % singleton sizes may still be dropped.
            inputSizePlaceholders = this.LayerAnalyzer.Inputs.Meta;
            inputSizes = cellfun(@(X)size(X), inputSizePlaceholders, UniformOutput=false);
        end        

        function outputSize = get.OutputSize(this)
            outputSize = this.LayerAnalyzer.Outputs.Size;
        end
        
        function tf = dltTensorHasLayout11CN(this, inputNum)
            sz = this.InputLayerSizes{inputNum};
            tf = numel(sz)==3 && isequal(sz(1:2), [1 1]);
        end
        function [nodeProto,nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, isForward)
            import nnet.internal.cnn.onnx.*;
            permutation = [];
            if isequal(inputTensorLayout,'snc')
                if isForward
                    permutation = [1 2 0];
                else
                    permutation = [2 0 1];
                end
            elseif isequal(inputTensorLayout,'snch')
                if isForward
                    permutation = [1 2 3 0];
                else
                    permutation = [3 0 1 2];
                end
            elseif isequal(inputTensorLayout,'snchw')
                if isForward
                    permutation = [1 2 3 4 0];
                else
                    permutation = [4 0 1 2 3];
                end
            elseif isequal(inputTensorLayout,'snchwd')
                if isForward
                    permutation = [1 2 3 4 5 0];
                else
                    permutation = [5 0 1 2 3 4];
                end
            end
            if ~isempty(permutation)
                [nodeProto,nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, permutation);
            end
        end
            
        function [nodeProto,nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, permutation)
            % Function to append ONNX Transpose Operator to the end of
            % Node Proto
            import nnet.internal.cnn.onnx.*;
            inputOperatorName           = mapTensorNames(this, {nextOperatorInput}, TensorNameMap);
            transposeONNXName           = 'transpose';
            transposeONNXName           = makeUniqueName({nodeProto.name}, transposeONNXName);
            [transposeNode, ~] = createNodeProto(this, 'Transpose', transposeONNXName,...
                inputOperatorName, {transposeONNXName}, permutation);
            nodeProto(end+1)            = transposeNode;
            nextOperatorInput           = transposeONNXName;
        end
        
        function [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = updateNetworkOutputForDanglingLayers(this, nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap)
            
            % In this function, output tensor information was collected
            % from current layer (dangling layer). For each output
            % tensor found, get its DLT size from the layerAnalyzer's
            % Outputs table, and get its ONNX layout from the
            % TensorLayoutMap. 
            
            if ~this.IsDanglingLayer
                return
            end
            
            import nnet.internal.cnn.onnx.*
            
            % Collect information from current node. Determine the network output
            % tensor size.
            ONNXName                 = this.NNTLayer.Name;
            inputTensorName         = this.mapNodeProtoNametoTensor({ONNXName}, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorName{1});
            dltOutputSize            = this.OutputSize{1};
            
            newNode = [];
            
            switch inputTensorLayout
                case {'nch', 'nchw', 'nchwd'}
                    if isDLTTensorFlat(this, dltOutputSize)
                        % If dltOutputSize is flattened, add a Flatten(1) operator
                        flattenNodeName        = [inputTensorName{1} '_Flatten'];
                        flattenNodeName        = makeUniqueName({nodeProto.name}, flattenNodeName);
                        FlattenNode            = NodeProto;
                        FlattenNode.op_type    = 'Flatten';
                        FlattenNode.name       = flattenNodeName;
                        FlattenNode.output     = {flattenNodeName};
                        FlattenNode.attribute  = makeAttributeProto('axis', 'INT', 1);
                        
                        % Output a Flatten(1) operator
                        FlattenNode.input      = inputTensorName(1);
                        outputTensorName       = flattenNodeName;
                        newNode                = FlattenNode;
                        
                        outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize(end))];	% {N C}
                    else
                        % If dltOutputSize is not flattened, create output tensor
                        % directly
                        outputTensorName       = inputTensorName{1};
                        outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize([end 1:(end-1)]))];	% {N C H W} or {N C H W D}
                    end
                    
                case 'snc'
                    if ~isscalar(dltOutputSize)
                        error(message('nnet_cnn_onnx:onnx:FlatOutputRequired', this.NNTLayer.Name));
                    end
                    
                    outputTensorName       = inputTensorName{1};
                    outputTensorSize       = [{1 this.BatchSizeToExport}, num2cell(dltOutputSize)];	% {1 N C}
                case 'nc'
                    if ~isscalar(dltOutputSize)
                        error(message('nnet_cnn_onnx:onnx:FlatOutputRequired', this.NNTLayer.Name));
                    end
                    
                    outputTensorName       = inputTensorName{1};
                    outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize)];	% {N C}
                case '1nc'
                    if ~isscalar(dltOutputSize)
                        error(message('nnet_cnn_onnx:onnx:FlatOutputRequired', this.NNTLayer.Name));
                    end

                    flattenNodeName        = [inputTensorName{1} '_Flatten'];
                    flattenNodeName        = makeUniqueName({nodeProto.name}, flattenNodeName);
                    FlattenNode            = NodeProto;
                    FlattenNode.op_type    = 'Flatten';
                    FlattenNode.name       = flattenNodeName;
                    FlattenNode.output     = {flattenNodeName};
                    FlattenNode.attribute  = makeAttributeProto('axis', 'INT', 2);

                    % Output a Flatten(2) operator
                    FlattenNode.input      = inputTensorName(1);
                    outputTensorName       = flattenNodeName;
                    newNode                = FlattenNode;
               
                    outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize)];	% {N C}
                    
                case {'snch','snchw', 'snchwd'}
                    % Create output tensor directly
                    outputTensorName        = inputTensorName{1};
                    outputTensorSize        = [{1 this.BatchSizeToExport}, num2cell(dltOutputSize([end 1:(end-1)]))];	%{1 N C H}, {1 N C H W}, or {1 N C H W D}
                    
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXOutputLayout', this.NNTLayer.Name));
            end
            
            % Set outputs
            nodeProto               = [nodeProto newNode];
            netOutputs              = makeValueInfoProtoFromDimensions(outputTensorName, TensorProto_DataType.FLOAT, outputTensorSize);
            networkOutputs          = [networkOutputs, netOutputs];
        end
        

        
        function tensorName = mapNodeProtoNametoTensor(this, namesCell, TensorNameMap)
            
            if isa(this,'nnet.internal.cnn.onnx.exporter.ConverterForMaxPoolingLayers') && ...
                    numel(this.OutputSize) == 3
                % Edge case: layerGraph/dlnetwork ends with
                % maxPooling2dLayer('HasUnpoolingOutputs',true)
                namesCell{1}    = legalizeNNTName(this,namesCell{1});
                tensorName      = {[namesCell{1} '_out']};
            else
                tensorName   = mapTensorNames(this, namesCell, TensorNameMap);
            end
            
        end
        
        function tf = isDLTTensorFlat(~, tensorSize)
            % True for a scalar, or a vector of length N whose first N-1 components are 1.
            N  = numel(tensorSize);
            tf = N==1 || isequal(tensorSize(1:(N-1)), ones(1,N-1));
        end
        
        function nodeProto = insertNodeBeforeSoftmax(~, nodeProto, newNode, softmaxNodeName, newSoftmaxAxis)
            % Insert a node before the softmax.
            import nnet.internal.cnn.onnx.*
            
            % In a multi-output network, the softmax node might not
            % be last in the nodeProto array, so traverse the nodes in reverse order to
            % find the correct node.
            for i=numel(nodeProto):-1:1
                nodeName = nodeProto(i).name;
                if strcmp(nodeName, softmaxNodeName)
                    smNode = nodeProto(i);
                    break
                end
            end
            
            if nargin > 4
                if ~isequal(smNode.attribute(1).name, 'axis')
                    error(message('nnet_cnn_onnx:onnx:FirstSoftmaxAxisAttribute'));
                end
                smNode.attribute(1) = makeAttributeProto('axis', 'INT', newSoftmaxAxis);
            end
            
            % Rewire connections
            newNode.input(1) = smNode.input;
            smNode.input	= {newNode.name};
            
            % Insert node
            nodeProtosBeforeSoftmax = nodeProto(1:i-1);
            nodeProtosAfterSoftmax = nodeProto(i+1:end);
            nodeProto       = [nodeProtosBeforeSoftmax newNode smNode nodeProtosAfterSoftmax];
        end

        function [nodeProto, initializers] = createNodeProto(this, op_type, varargin)
            switch op_type
                case 'Add'
                    fun = @nnet.internal.cnn.onnx.exporter.createAddNode;
                case 'AveragePool'
                    fun = @nnet.internal.cnn.onnx.exporter.createAveragePoolNode;
                case 'BatchNormalization'
                    fun = @nnet.internal.cnn.onnx.exporter.createBatchNormalizationNode;
                case 'Cast'
                    fun = @nnet.internal.cnn.onnx.exporter.createCastNode;
                case 'Clip'
                    fun = @nnet.internal.cnn.onnx.exporter.createClipNode;
                case 'Concat'
                    fun = @nnet.internal.cnn.onnx.exporter.createConcatNode;
                case 'Dropout'
                    fun = @nnet.internal.cnn.onnx.exporter.createDropoutNode;
                case 'Elu'
                    fun = @nnet.internal.cnn.onnx.exporter.createEluNode;
                case 'Flatten'
                    fun = @nnet.internal.cnn.onnx.exporter.createFlattenNode;
                case 'Gather'
                    fun = @nnet.internal.cnn.onnx.exporter.createGatherNode;
                case 'GlobalMaxPool'
                    fun = @nnet.internal.cnn.onnx.exporter.createGlobalMaxPoolNode;
                case 'GRUForward'
                    fun = @nnet.internal.cnn.onnx.exporter.createGRUForwardNode;
                case 'InstanceNormalization'
                    fun = @nnet.internal.cnn.onnx.exporter.createInstanceNormalizationNode;
                case 'LeakyRelu'
                    fun = @nnet.internal.cnn.onnx.exporter.createLeakyReluNode;
                case 'LRN'
                    fun = @nnet.internal.cnn.onnx.exporter.createLRNNode;
                case 'LSTMBidirectional'
                    fun = @nnet.internal.cnn.onnx.exporter.createLSTMBidirectionalNode;
                case 'LSTMForward'
                    fun = @nnet.internal.cnn.onnx.exporter.createLSTMForwardNode;
                case 'MaxUnPool'
                    fun = @nnet.internal.cnn.onnx.exporter.createMaxUnPoolNode;
                case 'MaxPool'
                    fun = @nnet.internal.cnn.onnx.exporter.createMaxPoolNode;
                case 'MaxRoiPool'
                    fun = @nnet.internal.cnn.onnx.exporter.createMaxRoiPoolNode;
                case 'Mul'
                    fun = @nnet.internal.cnn.onnx.exporter.createMulNode;
                case 'Pad'
                    fun = @nnet.internal.cnn.onnx.exporter.createPadNode;
                case 'Rescale'
                    fun = @nnet.internal.cnn.onnx.exporter.createRescaleNodes;
                case 'Softmax'
                    fun = @nnet.internal.cnn.onnx.exporter.createSoftmaxNode;
                case 'SISO'
                    fun = @nnet.internal.cnn.onnx.exporter.createSISONode;
                case 'Slice'
                    fun = @nnet.internal.cnn.onnx.exporter.createSliceNode;
                case 'SpaceToDepth'
                    fun = @nnet.internal.cnn.onnx.exporter.createSpaceToDepthNode;
                case 'Split'
                    fun = @nnet.internal.cnn.onnx.exporter.createSplitNode;
                case 'Squeeze'
                    fun = @nnet.internal.cnn.onnx.exporter.createSqueezeNode;
                case 'Transpose'
                    fun = @nnet.internal.cnn.onnx.exporter.createTransposeNode;
                case 'Unsqueeze'
                    fun = @nnet.internal.cnn.onnx.exporter.createUnsqueezeNode;
                case 'ZeroCenter'
                    fun = @nnet.internal.cnn.onnx.exporter.createZeroCenterNodes;
                case 'ZScore'
                    fun = @nnet.internal.cnn.onnx.exporter.createZScoreNodes;
                otherwise 
                    assert(false);
            end
            [nodeProto, initializers] = fun(this.OpsetVersion, varargin{:});
        end

        
    end
end

function flags = isdigit(c)
% Return true for every char in c that is a digit
flags = (c>='0' & c<='9');
end



