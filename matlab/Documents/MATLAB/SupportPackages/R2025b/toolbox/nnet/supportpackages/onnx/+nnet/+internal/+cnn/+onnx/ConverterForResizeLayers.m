classdef ConverterForResizeLayers < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert resize2dLayer or resize3dLayer into ONNX.
    
    % Copyright 2020-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForResizeLayers(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap, connections)
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            
            % Create the resize node.
            resizeNode         = NodeProto;
            resizeNode.name    = onnxName;
            resizeNode.output  = {onnxName};
            
            switch this.OpsetVersion 
                case num2cell(1:6)
                    resizeNode.op_type = 'Upsample';
                    resizeNode.input   = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
                    method = this.NNTLayer.Method;
                    if isa(this.NNTLayer, 'nnet.cnn.layer.Resize3DLayer')
                        error(message("nnet_cnn_onnx:onnx:CantExportLayer", class(this.NNTLayer)));
                    end
                    [scale, ~, outputTensorLayout] = iGetScaleValues(this);
                    if any(scale<1)
                        error(message("nnet_cnn_onnx:onnx:UnsupportedResizeScale"));
                    end
                    if strcmp(this.NNTLayer.GeometricTransformMode,'half-pixel')
                        warning(message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyExport", this.NNTLayer.Name, 'GeometricTransformMode',this.NNTLayer.GeometricTransformMode,this.OpsetVersion));
                    end
                    if strcmp(this.NNTLayer.NearestRoundingMode,'round')
                        warning(message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyExport", this.NNTLayer.Name, 'NearestRoundingMode',this.NNTLayer.NearestRoundingMode,this.OpsetVersion));
                    end
                    resizeNode.attribute = [...
                        makeAttributeProto('height_scale', 'FLOAT', scale(end-1)),...
                        makeAttributeProto('mode', 'STRING', method),...
                        makeAttributeProto('width_scale', 'FLOAT', scale(end))
                        ];
                    parameterInitializers = [];
                    nodeProto             = [nodeProto resizeNode];
                case num2cell(7:8)
                    resizeNode.op_type = 'Upsample';
                    resizeNode.input    = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
                    method = iConvertMethodToONNX(this.NNTLayer, this.OpsetVersion);
                    [scale, ~, outputTensorLayout] = iGetScaleValues(this);
                    if any(scale<1)
                        error(message("nnet_cnn_onnx:onnx:UnsupportedResizeScale"));
                    end
                    if strcmp(this.NNTLayer.GeometricTransformMode,'half-pixel')
                        warning(message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyExport", this.NNTLayer.Name, 'GeometricTransformMode',this.NNTLayer.GeometricTransformMode,this.OpsetVersion));
                    end
                    if strcmp(this.NNTLayer.NearestRoundingMode,'round')
                        warning(message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyExport", this.NNTLayer.Name, 'NearestRoundingMode',this.NNTLayer.NearestRoundingMode,this.OpsetVersion));
                    end
                    [resizeNode,parameterInitializers] = iFillResizeNodeAttributes(resizeNode, scale, method);
                    nodeProto                          = [nodeProto resizeNode];
                case 9
                    resizeNode.op_type = 'Upsample';
                    scalesName = [onnxName '_scales'];
                    method = iConvertMethodToONNX(this.NNTLayer, this.OpsetVersion);
                    resizeNode.input   = mapTensorNames(this, {this.InputLayerNames{1},scalesName}, TensorNameMap);
                    [scale, ~, outputTensorLayout] = iGetScaleValues(this);
                    if any(scale<1)
                        error(message("nnet_cnn_onnx:onnx:UnsupportedResizeScale"));
                    end
                    if strcmp(this.NNTLayer.GeometricTransformMode,'half-pixel')
                        warning(message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyExport", this.NNTLayer.Name, 'GeometricTransformMode',this.NNTLayer.GeometricTransformMode,this.OpsetVersion));
                    end
                    if strcmp(this.NNTLayer.NearestRoundingMode,'round')
                        warning(message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyExport", this.NNTLayer.Name, 'NearestRoundingMode',this.NNTLayer.NearestRoundingMode,this.OpsetVersion));
                    end
                    [resizeNode,parameterInitializers] = iFillResizeNodeAttributes(resizeNode, scale, method);
                    nodeProto                          = [nodeProto resizeNode];
                case 10
                    resizeNode.op_type = 'Resize';
                    scalesName = [onnxName '_scales'];
                    method = iConvertMethodToONNX(this.NNTLayer, this.OpsetVersion);
                    resizeNode.input   = mapTensorNames(this, {this.InputLayerNames{1},scalesName}, TensorNameMap);
                    [scale, ~, outputTensorLayout] = iGetScaleValues(this);
                    if strcmp(this.NNTLayer.GeometricTransformMode,'half-pixel')
                        warning(message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyExport", this.NNTLayer.Name, 'GeometricTransformMode',this.NNTLayer.GeometricTransformMode,this.OpsetVersion));
                    end
                    if strcmp(this.NNTLayer.NearestRoundingMode,'round')
                        warning(message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyExport", this.NNTLayer.Name, 'NearestRoundingMode',this.NNTLayer.NearestRoundingMode,this.OpsetVersion));
                    end
                    [resizeNode,parameterInitializers] = iFillResizeNodeAttributes(resizeNode, scale, method);
                    nodeProto                          = [nodeProto resizeNode];
                case num2cell(11:17)
                    % For opset version 11-17.
                    resizeNode.op_type = 'Resize';
                    method = iConvertMethodToONNX(this.NNTLayer, this.OpsetVersion);
                    coordinateTransformMode = iConvertGeometricTransformToONNX(this.NNTLayer.GeometricTransformMode);
                    if ~isempty(this.NNTLayer.OutputSize)
                        % resize to a specific target size
                        [resizeNodeProtos, parameterInitializers, outputTensorLayout] = createOperatorsForOutputSizeOption(this, coordinateTransformMode, method, resizeNode, TensorNameMap);
                        nodeProto = [nodeProto, resizeNodeProtos];
                    elseif this.NNTLayer.EnableReferenceInput
                        % resize in reference of a dlarray size
                        [resizeNodeProtos, parameterInitializers, outputTensorLayout] = createOperatorsForReferenceOption(this, coordinateTransformMode, method, resizeNode, connections, TensorNameMap);
                        nodeProto = [nodeProto, resizeNodeProtos];
                    else
                        % resize based on scales
                        [scale, isUpsample, outputTensorLayout] = iGetScaleValues(this);
                        nearestMode = iConvertNearestModeToONNX(this.NNTLayer.NearestRoundingMode, isUpsample);
                        roiName = [onnxName '_roi'];
                        t1              = TensorProto;
                        t1.name         = roiName;
                        t1.data_type	= TensorProto_DataType.FLOAT;
                        t1.raw_data     = rawData([]);
                        t1.dims         = dimVector(numel([]),1);
                        scalesName = [onnxName '_scales'];
                        t2              = TensorProto;
                        t2.name         = scalesName;
                        t2.data_type	= TensorProto_DataType.FLOAT;
                        t2.raw_data     = rawData(single(scale));
                        t2.dims         = dimVector(numel(scale),1);
                        resizeNode.input = mapTensorNames(this, {this.InputLayerNames{1},roiName,scalesName}, TensorNameMap);
                        resizeNode.attribute = [...
                            makeAttributeProto('mode', 'STRING', method),...
                            makeAttributeProto('coordinate_transformation_mode', 'STRING', coordinateTransformMode),...
                            makeAttributeProto('cubic_coeff_a', 'FLOAT', -0.75),...
                            makeAttributeProto('exclude_outside', 'INT', 0),...
                            makeAttributeProto('extrapolation_value', 'FLOAT', 0),...
                            makeAttributeProto('nearest_mode', 'STRING', nearestMode)
                            ];
                        nodeProto               = [nodeProto resizeNode];
                        parameterInitializers   = [t1, t2];
                    end     
                otherwise % opset >= 18
                    % 'roi' and 'scales' are now optional. 'roi' is only used 
                    % when coordinate_transformation_mode=tf_crop_and_resize. 
                    % Exactly one of 'scales' or 'sizes' must be set. The other 
                    % may be specified as an empty string.
                    resizeNode.op_type = 'Resize';
                    method = iConvertMethodToONNX(this.NNTLayer, this.OpsetVersion);
                    coordinateTransformMode = iConvertGeometricTransformToONNX(this.NNTLayer.GeometricTransformMode);
                    if ~isempty(this.NNTLayer.OutputSize)
                        [targetSize, isUpsample, outputTensorLayout, ~] = iGetSizeValues(this);
                        sizesName = [onnxName '_sizes'];
                        resizeNode.input = mapTensorNames(this, {this.InputLayerNames{1},'','',sizesName}, TensorNameMap);
                        nearestMode = iConvertNearestModeToONNX(this.NNTLayer.NearestRoundingMode, isUpsample);
                        resizeNode.attribute = [...
                            makeAttributeProto('mode', 'STRING', method),...
                            makeAttributeProto('coordinate_transformation_mode', 'STRING', coordinateTransformMode),...
                            makeAttributeProto('cubic_coeff_a', 'FLOAT', -0.75),...
                            makeAttributeProto('exclude_outside', 'INT', 0),...
                            makeAttributeProto('extrapolation_value', 'FLOAT', 0),...
                            makeAttributeProto('nearest_mode', 'STRING', nearestMode),...
                            makeAttributeProto('keep_aspect_ratio_policy', 'STRING', 'stretch')
                            ];
                        % add axes attribute to the resizeNode
                        if isa(this.NNTLayer, 'nnet.cnn.layer.Resize2DLayer')
                            resizeNode.attribute = [resizeNode.attribute,...
                                makeAttributeProto('axes', 'INTS', [2,3])];
                            % targetSize CSS->SS
                            targetSize = targetSize(2:3);
                        else
                            resizeNode.attribute = [resizeNode.attribute,...
                                makeAttributeProto('axes', 'INTS', [2,3,4])];
                            % targetSize CSSS->SSS
                            targetSize = targetSize(2:4);
                        end
                        % create tensorProto for "sizes" input
                        t1              = TensorProto;
                        t1.name         = sizesName;
                        t1.data_type	= TensorProto_DataType.INT64;
                        t1.raw_data     = rawData(int64(targetSize));
                        t1.dims         = dimVector(numel(targetSize),1);   
                        parameterInitializers   = t1;
                        nodeProto               = [nodeProto resizeNode];
                    elseif this.NNTLayer.EnableReferenceInput
                        % resize in reference of a dlarray size
                        [resizeNodeProtos, parameterInitializers, outputTensorLayout] = createOperatorsForReferenceOption(this, coordinateTransformMode, method, resizeNode, connections, TensorNameMap);
                        nodeProto = [nodeProto, resizeNodeProtos];
                    else
                        scalesName = [onnxName '_scales'];
                        [scale, isUpsample, outputTensorLayout] = iGetScaleValues(this);
                        resizeNode.input = mapTensorNames(this, {this.InputLayerNames{1},'',scalesName}, TensorNameMap);
                        nearestMode = iConvertNearestModeToONNX(this.NNTLayer.NearestRoundingMode, isUpsample);
                        resizeNode.attribute = [...
                            makeAttributeProto('mode', 'STRING', method),...
                            makeAttributeProto('coordinate_transformation_mode', 'STRING', coordinateTransformMode),...
                            makeAttributeProto('cubic_coeff_a', 'FLOAT', -0.75),...
                            makeAttributeProto('exclude_outside', 'INT', 0),...
                            makeAttributeProto('extrapolation_value', 'FLOAT', 0),...
                            makeAttributeProto('nearest_mode', 'STRING', nearestMode)
                            ];
                        % create tensorProto for "scales" input
                        t1              = TensorProto;
                        t1.name         = scalesName;
                        t1.data_type	= TensorProto_DataType.FLOAT;
                        t1.raw_data     = rawData(single(scale));
                        t1.dims         = dimVector(numel(scale),1);
                        resizeNode.input = mapTensorNames(this, {this.InputLayerNames{1},'',scalesName}, TensorNameMap);
                        parameterInitializers   = t1;
                        nodeProto               = [nodeProto resizeNode];
                    end
            end
            networkInputs           = [];
            networkOutputs          = [];
            
            TensorNameMap(this.NNTLayer.Name) = onnxName;
            % Update layout map
            TensorLayoutMap(onnxName) = outputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

function method  = iConvertMethodToONNX(layer,version)
if strcmp(layer.Method,'bilinear')
    method = 'linear';
elseif strcmp(layer.Method,'trilinear')
    % Currently, onnxruntime does not support linear method for 5D data.
    % Switching method to nearest.
    % https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/tensor/upsample.h#L278
    warning(message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyExport", layer.Name,'Method',layer.Method,version));
    method = 'nearest';
else
    method = 'nearest';
end
end

function [nodeProtos, inits, outputTensorLayout] = createOperatorsForOutputSizeOption(this,coordinateTransformMode, method, resizeNode, TensorNameMap)
    % for opset 11-17 
    % input -> shape -> slice (get B) -> concat (BCSS) -> resize
    % initializer(CSS)                ->
    import nnet.internal.cnn.onnx.*
    [targetSize, isUpsample, ~, ~] = iGetSizeValues(this);
    % CSS dimension initializer
    onnxName = resizeNode.name;
    cssName = [onnxName, '_css'];
    cssInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(cssName, ...
        3, targetSize, TensorProto_DataType.INT64);
    % shape node
    inputShapeNode = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
    shapeName = [onnxName, '_input_shape'];
    [shapeNode, ~] = createNodeProto(this, 'SISO', 'Shape',...
        shapeName, inputShapeNode, {shapeName});
    % slice node
    sliceName = [onnxName, '_input_slice'];
    [sliceNode, sliceInits] = createNodeProto(this, 'Slice',...
        sliceName, {shapeName}, {sliceName},0,1);
    % concat node
    concatName = [onnxName, '_input_concat'];
    [concatNode, ~] = createNodeProto(this, 'Concat',...
        concatName, {sliceName, cssName}, {concatName}, 0);
    % create empty tensors for roi and scales
    roiName = [onnxName '_roi'];
    t1              = TensorProto;
    t1.name         = roiName;
    t1.data_type	= TensorProto_DataType.FLOAT;
    t1.raw_data     = rawData([]);
    t1.dims         = dimVector(numel([]),1);
    scalesName = [onnxName '_scales'];
    t2              = TensorProto;
    t2.name         = scalesName;
    t2.data_type	= TensorProto_DataType.FLOAT;
    t2.raw_data     = rawData([]);
    t2.dims         = dimVector(numel([]),1);
    % resize node
    nearestMode = iConvertNearestModeToONNX(this.NNTLayer.NearestRoundingMode, isUpsample); 
    resizeNode.input = mapTensorNames(this, {this.InputLayerNames{1},roiName,scalesName,concatName}, TensorNameMap);
    resizeNode.attribute = [...
        makeAttributeProto('mode', 'STRING', method),...
        makeAttributeProto('coordinate_transformation_mode', 'STRING', coordinateTransformMode),...
        makeAttributeProto('cubic_coeff_a', 'FLOAT', -0.75),...
        makeAttributeProto('exclude_outside', 'INT', 0),...
        makeAttributeProto('extrapolation_value', 'FLOAT', 0),...
        makeAttributeProto('nearest_mode', 'STRING', nearestMode)
        ];

    nodeProtos = [shapeNode, sliceNode, concatNode, resizeNode];
    inits = [cssInit, sliceInits, t1, t2];
    if isa(this.NNTLayer, 'nnet.cnn.layer.Resize2DLayer')
        outputTensorLayout = 'nchw';
    else
        outputTensorLayout = 'nchwd';
    end
end

function [nodeProtos, inits, outputTensorLayout] = createOperatorsForReferenceOption(this, coordinateTransformMode, method, resizeNode, connections, TensorNameMap)
    % ref dlarray -> shape -> slice (get SS or SSS) ->
    % input       -> shape -> slice (get BC)        -> concat (BCSS) -> resize

    import nnet.internal.cnn.onnx.*
    onnxName = resizeNode.name;
    % get reference input name
    referenceInputName = [this.NNTLayer.Name, '/ref'];
    referenceInput = connections.Source(find(strcmp(connections.Destination, referenceInputName)));
    referenceInputONNXName = mapTensorNames(this, referenceInput(1), TensorNameMap);
    % ref shape node
    refShapeName = [referenceInputONNXName{1},'_',onnxName, '_shape'];
    [refShapeNode, ~] = createNodeProto(this, 'SISO', 'Shape',...
        refShapeName, referenceInputONNXName, {refShapeName});
    % ref slice node
    refSliceName = [referenceInputONNXName{1},'_',onnxName, '_slice'];
    [refSliceNode, refSliceInits] = createNodeProto(this, 'Slice',...
        refSliceName, {refShapeName}, {refSliceName},2,5);
    % input shape node
    resizeShapeName = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
    inputShapeName = [onnxName, '_input_shape'];
    [inputShapeNode, ~] = createNodeProto(this, 'SISO', 'Shape',...
        inputShapeName, resizeShapeName, {inputShapeName});
    % input slice node
    inputSliceName = [onnxName, '_input_slice'];
    [inputSliceNode, inputSliceInits] = createNodeProto(this, 'Slice',...
        inputSliceName, {inputShapeName}, {inputSliceName},0,2);
    % concat node
    concatName = [onnxName, '_input_concat'];
    [concatNode, ~] = createNodeProto(this, 'Concat',...
        concatName, {inputSliceName, refSliceName}, {concatName}, 0);
    % create empty tensors for roi and scales
    roiName = [onnxName '_roi'];
    t1              = TensorProto;
    t1.name         = roiName;
    t1.data_type	= TensorProto_DataType.FLOAT;
    t1.raw_data     = rawData([]);
    t1.dims         = dimVector(numel([]),1);
    scalesName = [onnxName '_scales'];
    t2              = TensorProto;
    t2.name         = scalesName;
    t2.data_type	= TensorProto_DataType.FLOAT;
    t2.raw_data     = rawData([]);
    t2.dims         = dimVector(numel([]),1);
    % resize node
    [~, isUpsample, outputTensorLayout, ~] = iGetSizeValues(this);
    nearestMode = iConvertNearestModeToONNX(this.NNTLayer.NearestRoundingMode, isUpsample);
    resizeNode.input = mapTensorNames(this, {this.InputLayerNames{1},roiName, scalesName, concatName}, TensorNameMap);
    resizeNode.attribute = [...
        makeAttributeProto('mode', 'STRING', method),...
        makeAttributeProto('coordinate_transformation_mode', 'STRING', coordinateTransformMode),...
        makeAttributeProto('cubic_coeff_a', 'FLOAT', -0.75),...
        makeAttributeProto('exclude_outside', 'INT', 0),...
        makeAttributeProto('extrapolation_value', 'FLOAT', 0),...
        makeAttributeProto('nearest_mode', 'STRING', nearestMode)
        ];
    nodeProtos = [refShapeNode, refSliceNode, inputShapeNode,inputSliceNode, concatNode, resizeNode];
    inits = [refSliceInits, inputSliceInits, t1, t2];
end

function [targetSize, isUpsample, outputTensorLayout, inputTensorSize] = iGetSizeValues(this)
    % targetSize returns in CSS or CSSS dimension. Omit batch as there's no
    % condition we need B dimension info. 
    inputTensorSize = this.InputLayerSizes{1};
    % get the output size from layer output
    outputSize = this.NNTLayer.OutputSize;
    if isa(this.NNTLayer, 'nnet.cnn.layer.Resize2DLayer')
        if this.NNTLayer.EnableReferenceInput
            % get the output size from reference input if enableReferenceInput
            outputSize = this.InputLayerSizes{2}(1:2);
        end
        outputTensorLayout = 'nchw';
        inputImageSize  = inputTensorSize(1:2); 
        if ~any(isnan(outputSize))
            targetSize = [inputTensorSize(3), outputSize];
            isUpsample = all(outputSize./inputImageSize);
        else
            idx = find(~isnan(outputSize));
            % scale is calculated with double precision. 
            % suggest user to fill resize2DLayer OutputSize for all dimension 
            % if user reports that output size mismatches 
            % between exported resize operation and resize2DLayer
            scale = outputSize(idx)/inputImageSize(idx);
            targetSize = [inputTensorFullSize(3), floor(scale*inputTensorFullSize(1:2))];
            isUpsample = scale >= 1;
        end
    elseif isa(this.NNTLayer, 'nnet.cnn.layer.Resize3DLayer')
        outputTensorLayout = 'nchwd';
        inputImageSize  = inputTensorSize(1:3); 
        if this.NNTLayer.EnableReferenceInput
            % get the output size from reference input if enableReferenceInput
            outputSize = this.InputLayerSizes{2}(1:3);
        end
        if ~any(isnan(outputSize))
            targetSize = [inputTensorSize(4), outputSize];
            isUpsample = all(outputSize./inputImageSize);
        else
            idx = find(~isnan(outputSize));
            % scale is calculated with double precision. 
            % suggest user to fill resize3DLayer OutputSize for all dimension 
            % if user reports that output size mismatches 
            % between exported resize operation and resize3DLayer
            scale = outputSize(idx)/inputImageSize(idx);
            targetSize = [inputTensorFullSize(4), floor(scale*inputTensorFullSize(1:3))];
            isUpsample = scale >= 1;
        end
    else
        error(message("nnet_cnn_onnx:onnx:UnexpectedLayer", 'nnet.cnn.layer.Resize2DLayer', 'nnet.cnn.layer.Resize3DLayer', class(this.NNTLayer)));
    end
end
function [scale, isUpsample, outputTensorLayout, inputTensorSize] = iGetScaleValues(this)
scale = [];
if isa(this.NNTLayer, 'nnet.cnn.layer.Resize2DLayer')
    outputTensorLayout = 'nchw';
    inputTensorSize = this.InputLayerSizes{1};
    inputImageSize  = inputTensorSize(1:2); 
    if ~isempty(this.NNTLayer.Scale)
        scale = [1 1 this.NNTLayer.Scale];
    elseif ~isempty(this.NNTLayer.OutputSize)
        outputSize = this.NNTLayer.OutputSize;
        if ~any(isnan(outputSize))
            scale = [1 1 outputSize./inputImageSize];
        else
            idx = find(~isnan(outputSize));
            scale = [1 1 repelem(outputSize(idx)/inputImageSize(idx),2)];
        end
    elseif this.NNTLayer.EnableReferenceInput
        % Get the output size from the reference port
        outputSize = this.InputLayerSizes{2}(1:2);
        scale = [1 1 outputSize./inputImageSize];
    end
    inputTensorSize = inputTensorSize([3 1 2]); % CHW
elseif isa(this.NNTLayer, 'nnet.cnn.layer.Resize3DLayer')
    outputTensorLayout = 'nchwd';
    inputTensorSize = this.InputLayerSizes{1};    
    inputImageSize  = inputTensorSize(1:3);
    if ~isempty(this.NNTLayer.Scale)
        scale = [1 1 this.NNTLayer.Scale];
    elseif ~isempty(this.NNTLayer.OutputSize)
        outputSize = this.NNTLayer.OutputSize;
        if ~any(isnan(outputSize))
            scale = [1 1 outputSize./inputImageSize];
        else
            idx = find(~isnan(outputSize));
            scale = [1 1 repelem(outputSize(idx)/inputImageSize(idx),3)];
        end  
    elseif this.NNTLayer.EnableReferenceInput
        % Get the output size from the reference port
        outputSize = this.InputLayerSizes{2}(1:3);
        scale = [1 1 outputSize./inputImageSize];
    end
    inputTensorSize = inputTensorSize([4 1 2 3]); % CHWD
else
    error(message("nnet_cnn_onnx:onnx:UnexpectedLayer", 'nnet.cnn.layer.Resize2DLayer', 'nnet.cnn.layer.Resize3DLayer', class(this.NNTLayer)));
end
isUpsample = all(scale(3:end)>=1);
end

function [resizeNode,parameterInitializers] = iFillResizeNodeAttributes(resizeNode, scale, method)
import nnet.internal.cnn.onnx.*
if isscalar(resizeNode.input)
    resizeNode.attribute = [...
        makeAttributeProto('scales', 'FLOATS', scale),...
        makeAttributeProto('mode', 'STRING', method)
        ];
    parameterInitializers   = [];
else
    resizeNode.attribute = [...
        nnet.internal.cnn.onnx.makeAttributeProto('mode', 'STRING', method)
        ];
    
    % Make parameter for scales.
    t1              = TensorProto;
    t1.name         = [resizeNode.name '_scales'];
    t1.data_type	= TensorProto_DataType.FLOAT;
    t1.raw_data     = rawData(single(scale));
    t1.dims         = dimVector(numel(scale),1);
    
    parameterInitializers   = t1;
end
end

function val  = iConvertGeometricTransformToONNX(val)
if strcmp(val,'half-pixel')
    val = 'half_pixel';
end
end

function val  = iConvertNearestModeToONNX(val, isUpsample)
if strcmp(val,'round')
    val = 'round_prefer_ceil';
else
    if isUpsample
        val = 'floor';
    else
        val = 'ceil';
    end
end
end