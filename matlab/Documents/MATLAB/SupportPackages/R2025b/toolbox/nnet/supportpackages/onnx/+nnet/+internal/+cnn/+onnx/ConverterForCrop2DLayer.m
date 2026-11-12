classdef ConverterForCrop2DLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a Crop2DLayer into ONNX
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForCrop2DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            if this.OpsetVersion > 12
                % the exported ONNX model could be inferenced with different
                % input size only if the ONNX OpsetVersion is 13 and above
                [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx13(this, nodeProto, TensorNameMap, TensorLayoutMap);
            else
                [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx12(this, nodeProto, TensorNameMap, TensorLayoutMap);
            end
        end
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx12(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            % Find Left, Top, Right, Bottom of resulting region
            InputLayerSize      = this.InputLayerSizes{1};
            XH                  = InputLayerSize(1);
            XW                  = InputLayerSize(2);
            ReferenceLayerSize  = this.InputLayerSizes{2};
            H                   = ReferenceLayerSize(1);
            W                   = ReferenceLayerSize(2);
            switch this.NNTLayer.Mode
                case 'centercrop'
                    assert(isequal(this.NNTLayer.Location, 'auto'), ...
                        message('nnet_cnn_onnx:onnx:CenterCropAuto'));
                    sz           = InputLayerSize;
                    outputSize	 = ReferenceLayerSize(1:2);
                    % Compare the following to nnet.internal.cnn.layer.util.Crop2DCenterCropStrategy
                    centerX      = floor(sz(1:2)/2 + 1);
                    centerWindow = floor(outputSize/2 + 1);
                    offset       = centerX - centerWindow + 1;
                    T            = offset(1);
                    L            = offset(2);
                case 'custom'
                    assert(isnumeric(this.NNTLayer.Location) && numel(this.NNTLayer.Location)==2,...
                        message('nnet_cnn_onnx:onnx:UnsupportedCropLocation', this.NNTLayer.Name)); % [L T]
                    T = this.NNTLayer.Location(2);
                    L = this.NNTLayer.Location(1);
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnsupportedCropMode', this.NNTLayer.Mode, this.NNTLayer.Name));
            end
            B = T + H - 1;
            R = L + W - 1;
            
            % Convert from retained region to borders (number of rows and cols to delete):
            LB = L - 1;
            TB = T - 1;
            RB = XW - R;
            BB = XH - B;
                      
            % RB or BB may be zero if there are no rows/cols to delete. 
            % In this case, end should be intmax (retain the entire
            % input along the axis)
            endR = -RB;
            if endR == 0
                endR = intmax;
            end
            
            endB = -BB;              
            if endB == 0
                endB = intmax;
            end

            % Create Slice node
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            sliceName               = onnxName;
            sliceInput              = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            sliceOutput             = {onnxName};
            starts                  = [0 0 TB LB];
            ends                    = [intmax intmax endB endR];
            [sliceNode, inits]      = createNodeProto(this, 'Slice', sliceName, sliceInput, sliceOutput, starts, ends);

            nodeProto(end+1)        = sliceNode;
            parameterInitializers   = inits;
            networkInputs           = [];
            networkOutputs          = [];
            
            % Update maps
            TensorNameMap(this.NNTLayer.Name) = onnxName;
            TensorLayoutMap(onnxName)         = 'nchw';
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx13(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            [onnxNodeName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxNodeName                = makeUniqueName({nodeProto.name}, onnxNodeName);
            inputTensorNames = mapTensorNames(this, this.InputLayerNames, TensorNameMap);

            % Find Left, Top, Right, Bottom of resulting region
            % input feature Shape node
            inputFeatureShapeNodeName = [onnxNodeName '_InputShape'];
            [inputFeatureShapeNode, ~] = createNodeProto(this, 'SISO', 'Shape', inputFeatureShapeNodeName, inputTensorNames(1), {inputFeatureShapeNodeName});
            % cast the input shape to be non-int type for division
            inputFeatureShapeFloatName = [onnxNodeName '_InputShapeFloat'];
            [inputFeatureShapeFloatNode, ~] = createNodeProto(this, 'Cast', inputFeatureShapeFloatName, ...
                {inputFeatureShapeNodeName}, {inputFeatureShapeFloatName}, 1);
            % ref feature Shape node
            refFeatureShapeNodeName = [onnxNodeName '_RefShape'];
            [refFeatureShapeNode, ~] = createNodeProto(this, 'SISO', 'Shape', refFeatureShapeNodeName, inputTensorNames(2), {refFeatureShapeNodeName});
            % cast the ref shape to be non-int type for division
            refFeatureShapeFloatName = [onnxNodeName '_RefShapeFloat'];
            [refFeatureShapeFloatNode, ~] = createNodeProto(this, 'Cast', refFeatureShapeFloatName, ...
                {refFeatureShapeNodeName}, {refFeatureShapeFloatName}, 1);
            % ref feature's shape Split node
            refFeatureShapeSplitNodeName = [onnxNodeName '_RefShapeSplit'];
            refFeatureHNodeName = [onnxNodeName '_RefH'];
            refFeatureWNodeName = [onnxNodeName '_RefW'];
            refFeatureBNodeName = [onnxNodeName '_RefB'];
            refFeatureCNodeName = [onnxNodeName '_RefC'];
            [refFeatureShapeSplitNode, refFeatureShapeSplitInit] = createNodeProto(this, 'Split', refFeatureShapeSplitNodeName,...
                {refFeatureShapeFloatName}, {refFeatureBNodeName, refFeatureCNodeName,refFeatureHNodeName,refFeatureWNodeName},0,[1,1,1,1]);
            
            onnxInputNames = {
                inputFeatureShapeNodeName,...
                inputFeatureShapeFloatName,...
                refFeatureShapeNodeName,...
                refFeatureShapeFloatName,...
                refFeatureWNodeName,...
                refFeatureHNodeName,...
                [onnxNodeName '_L'],...
                [onnxNodeName '_T']};

            switch this.NNTLayer.Mode
                case 'centercrop'
                    assert(isequal(this.NNTLayer.Location, 'auto'), ...
                        message('nnet_cnn_onnx:onnx:CenterCropAuto'));
                    % create centerCrop nodes
                    [centerCropNodes, centerCropInits] = createCenterCropPreprocessingNodes(this, ...
                        onnxNodeName, onnxInputNames);
                case 'custom'
                    assert(isnumeric(this.NNTLayer.Location) && numel(this.NNTLayer.Location)==2,...
                        message('nnet_cnn_onnx:onnx:UnsupportedCropLocation', this.NNTLayer.Name)); % [L T]
                    T = this.NNTLayer.Location(2)-1;
                    L = this.NNTLayer.Location(1)-1;
                    TInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(onnxInputNames{8}, ...
                        1, T, TensorProto_DataType.FLOAT);
                    LInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(onnxInputNames{7}, ...
                        1, L, TensorProto_DataType.FLOAT);
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnsupportedCropMode', this.NNTLayer.Mode, this.NNTLayer.Name));
            end
            [IndexProcessNodes, IndexProcessInits, indicesName] = createIndexProcessingNodes(this, ...
                        onnxNodeName, onnxInputNames);
            
            % Create Slice node
            sliceName               = onnxNodeName;
            sliceInput              = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            sliceOutput             = {onnxNodeName};
            % Create start node
            ZeroName = [onnxNodeName '_zero'];
            ZeroInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(ZeroName, ...
                1, 0, TensorProto_DataType.INT64);
            % starts
            startsName = [onnxNodeName '_starts'];
            [startsNode, ~] = createNodeProto(this, 'Concat', startsName,...
                {ZeroName,ZeroName,indicesName{2},indicesName{1}}, {startsName}, 0);
            
            IntmaxName = [onnxNodeName '_intmax'];
            IntmaxInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(IntmaxName, ...
                1, intmax, TensorProto_DataType.INT64);
            % ends
            endsName = [onnxNodeName '_ends'];
            [endsNode, ~] = createNodeProto(this, 'Concat', endsName,...
                {IntmaxName,IntmaxName,indicesName{4},indicesName{3}}, {endsName}, 0);

            [sliceNode, sliceInits]      = createNodeProto(this, 'Slice', sliceName, sliceInput, sliceOutput, startsName, endsName);
            
            if isequal(this.NNTLayer.Mode,'centercrop')
                nodeProto = [nodeProto, inputFeatureShapeNode, inputFeatureShapeFloatNode, ...
                    refFeatureShapeNode, refFeatureShapeFloatNode, ...
                    refFeatureShapeSplitNode, centerCropNodes,...
                    IndexProcessNodes, startsNode, endsNode, sliceNode];
                parameterInitializers   = [refFeatureShapeSplitInit, ...
                    centerCropInits, IndexProcessInits, ZeroInit, IntmaxInit, sliceInits];
            else
                nodeProto = [nodeProto, ...
                    refFeatureShapeNode, refFeatureShapeFloatNode, ...
                    refFeatureShapeSplitNode, IndexProcessNodes, startsNode,...
                    endsNode, sliceNode];
                parameterInitializers   = [refFeatureShapeSplitInit, ...
                TInit, LInit, IndexProcessInits, ZeroInit, IntmaxInit, sliceInits];
            end
            
            networkInputs           = [];
            networkOutputs          = [];
            
            % Update maps
            TensorNameMap(this.NNTLayer.Name) = onnxNodeName;
            TensorLayoutMap(onnxNodeName)         = 'nchw';
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end

        function [nodes, inits] = createCenterCropPreprocessingNodes(this, onnxNodeName, ...
                onnxInputNames)
            import nnet.internal.cnn.onnx.*
            % Compare the following to nnet.internal.cnn.layer.util.Crop2DCenterCropStrategy
            % centerX      = floor(float(inputFeatureShape)/2 + 1);
            % centerWindow = floor(float(outputFeatureShape)/2 + 1);
            % offset       = centerX - centerWindow;
            % T            = offset(1);
            % L            = offset(2);

            % input feature centerX Div node
            inputFeatureDivNodeName = [onnxNodeName '_InputDiv'];
            inputFeatureDivNodeSecName = [onnxNodeName '_InputDivSec'];
            inputFeatureDivNodeInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(inputFeatureDivNodeSecName, ...
                1, 2, TensorProto_DataType.FLOAT);
            [inputFeatureDivNode, ~] = createNodeProto(this, 'SISO', 'Div', inputFeatureDivNodeName,...
                {onnxInputNames{2},inputFeatureDivNodeSecName}, {inputFeatureDivNodeName});
            % input feature centerX Add node
            inputFeatureAddNodeName = [onnxNodeName '_InputAdd'];
            inputFeatureAddNodeSecName = [onnxNodeName '_InputAddSec'];
            inputFeatureAddNodeInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(inputFeatureAddNodeSecName, ...
                1, 1, TensorProto_DataType.FLOAT);
            [inputFeatureAddNode, ~] = createNodeProto(this, 'SISO', 'Add', inputFeatureAddNodeName, ...
                {inputFeatureDivNodeName,inputFeatureAddNodeSecName}, {inputFeatureAddNodeName});
            % input feature centerX Floor node
            inputFeatureFloorNodeName = [onnxNodeName '_InputFloor'];
            [inputFeatureFloorNode, ~] = createNodeProto(this, 'SISO', 'Floor', inputFeatureFloorNodeName, ...
                {inputFeatureAddNodeName}, {inputFeatureFloorNodeName});
            
            % ref feature centerWindow Div node
            refFeatureDivNodeName = [onnxNodeName '_RefDiv'];
            refFeatureDivNodeSecName = [onnxNodeName '_RefDivSec'];
            refFeatureDivNodeInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(refFeatureDivNodeSecName, ...
                1, 2, TensorProto_DataType.FLOAT);
            [refFeatureDivNode, ~] = createNodeProto(this, 'SISO', 'Div', refFeatureDivNodeName, ...
                {onnxInputNames{4},refFeatureDivNodeSecName}, {refFeatureDivNodeName});
            % ref feature centerWindow Add node
            refFeatureAddNodeName = [onnxNodeName '_RefAdd'];
            refFeatureAddNodeSecName = [onnxNodeName '_RefAddSec'];
            refFeatureAddNodeInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(refFeatureAddNodeSecName, ...
                1, 1, TensorProto_DataType.FLOAT);
            [refFeatureAddNode, ~] = createNodeProto(this, 'SISO', 'Add', refFeatureAddNodeName, ...
                {refFeatureDivNodeName,refFeatureAddNodeSecName}, {refFeatureAddNodeName});
            % ref feature centerWindow Floor node
            refFeatureFloorNodeName = [onnxNodeName '_RefFloor'];
            [refFeatureFloorNode, ~] = createNodeProto(this, 'SISO', 'Floor', refFeatureFloorNodeName, ...
                {refFeatureAddNodeName}, {refFeatureFloorNodeName});
            
            % offset Sub node
            offsetSubNodeName = [onnxNodeName '_OffsetSub'];
            [offsetSubNode, ~] = createNodeProto(this, 'SISO', 'Sub', offsetSubNodeName, ...
                {inputFeatureFloorNodeName,refFeatureFloorNodeName}, {offsetSubNodeName});

            % split node
            splitNodeName = [onnxNodeName '_Split'];
            BNodeName = [onnxNodeName '_SplitB'];
            CNodeName = [onnxNodeName '_SplitC'];
            [splitNode, splitInit] = createNodeProto(this, 'Split', splitNodeName, ...
                {offsetSubNodeName}, {BNodeName,CNodeName,onnxInputNames{8},onnxInputNames{7}},0,[1,1,1,1]);

            nodes = [inputFeatureDivNode,...
                inputFeatureAddNode, inputFeatureFloorNode, refFeatureDivNode,...
                refFeatureAddNode, refFeatureFloorNode, offsetSubNode, ...
                splitNode];
            inits = [inputFeatureDivNodeInit,inputFeatureAddNodeInit,refFeatureDivNodeInit,...
                refFeatureAddNodeInit, splitInit];
        end

        function [nodes, inits, indicesName] = createIndexProcessingNodes(this, ...
                        onnxNodeName, onnxInputNames)
            import nnet.internal.cnn.onnx.*
            %LB = L
            LBNodeName = [onnxNodeName '_LB'];
            [LBNode, ~] = createNodeProto(this, 'Cast', LBNodeName, onnxInputNames(7), {LBNodeName},7);
            % TB = T
            TBNodeName = [onnxNodeName '_TB'];
            [TBNode, ~] = createNodeProto(this, 'Cast', TBNodeName, onnxInputNames(8), {TBNodeName},7);
            % RB = L+W;
            RBFloatNodeName = [onnxNodeName '_RBFloat'];
            [RBFloatNode, ~] = createNodeProto(this, 'SISO', 'Add', RBFloatNodeName, {onnxInputNames{7},onnxInputNames{5}}, {RBFloatNodeName});
            RBNodeName = [onnxNodeName '_RB'];
            [RBNode, ~] = createNodeProto(this, 'Cast', RBNodeName, {RBFloatNodeName}, {RBNodeName},7);
            % BB = T+H;
            BBFloatNodeName = [onnxNodeName '_BBFloat'];
            [BBFloatNode, ~] = createNodeProto(this, 'SISO', 'Add', BBFloatNodeName, {onnxInputNames{8},onnxInputNames{6}}, {BBFloatNodeName});
            BBNodeName = [onnxNodeName '_BB'];
            [BBNode, ~] = createNodeProto(this, 'Cast', BBNodeName, {BBFloatNodeName}, {BBNodeName},7);

            nodes = [LBNode, TBNode, RBFloatNode, RBNode, BBFloatNode, BBNode];

            inits = [];
            indicesName = {...
                LBNodeName, ...
                TBNodeName,...
                RBNodeName,...
                BBNodeName};
        end
    end
end