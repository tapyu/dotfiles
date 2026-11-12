classdef ConverterForBiLSTMLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a bilstmLayer into ONNX

    % Supported IO as of 23a:
    % CBT(sequence) --> CBT
    % CBT(last) --> CB
    % CB(sequence) --> CB
    % CB(last) --> CB
    % Hidden and Cell inputs and outputs, when present, are always CB.

    % Copyright 2018-2025 The MathWorks, Inc.

    methods
        function this = ConverterForBiLSTMLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] = toOnnx(...
                this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            networkInputs           = [];
            networkOutputs          = [];
            existingNodeNames       = {nodeProto.name};
            [lstmNodeName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            lstmNodeName                = makeUniqueName(existingNodeNames, lstmNodeName);
            NNTLayer = this.NNTLayer;
            hasSequenceOutput = isequal(NNTLayer.OutputMode, 'sequence');

            inputTensorNames = mapTensorNames(this, this.InputLayerNames, TensorNameMap);

            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});

            % Unsqueeze if inputTensorLayout is 'CB'
            if inputTensorLayout == "nc"
                unsqueezeAxis                 = 0;
                unsqueezeName                 = [inputTensorNames{1} '_Unsqueeze'];
                unsqueezeName                 = makeUniqueName(existingNodeNames, unsqueezeName);
                unsqueezeInput                = {inputTensorNames{1}};
                unsqueezeOutput               = {unsqueezeName};
                [unsqueezeNode, unsqueezeInit]  = createNodeProto(this, 'Unsqueeze', unsqueezeName, unsqueezeInput, unsqueezeOutput, unsqueezeAxis);
                lstmInputTensorName           = unsqueezeName;
                hasSequenceOutput             = false;
            elseif inputTensorLayout == "snc"
                lstmInputTensorName = inputTensorNames{1};
                unsqueezeNode = [];
                unsqueezeInit = [];
            end

            % Set outputTensorLayout
            if hasSequenceOutput
                lstmOutputTensorLayout  = 'snc';
            else
                lstmOutputTensorLayout  = 'nc';
            end

            % Make the LSTM inputs
            lstmInput                   = {...
                lstmInputTensorName,... % X
                [lstmNodeName '_W'],...
                [lstmNodeName '_R'],...
                [lstmNodeName '_B'],...
                '',...                      % sequence_Lens
                [lstmNodeName '_initial_h'],...
                [lstmNodeName '_initial_c']
                };
            if NNTLayer.HasStateInputs
                warning(message('nnet_cnn_onnx:onnx:UnsupportedRNNStateInputs', this.NNTLayer.Name))
                lstmInput(6:7) = inputTensorNames(2:3);
            end

            % Create nodes to transform the state inputs if necessary.
            [StateInputPreprocessingNodes, StateInputPreprocessingInits, lstmInput] = createStateInputPreprocessingNodes(...
                this, NNTLayer, lstmNodeName, inputTensorNames, lstmInput, existingNodeNames, TensorNameMap, TensorLayoutMap);

            lstmInput = mapTensorNames(this, lstmInput(:)', TensorNameMap);

            % Make the output name appear as either the sequence output or
            % the 'last' output as appropriate. Include state outputs if
            % necessary.
            if NNTLayer.HasStateOutputs
                lstmOutputTensorName    = [lstmNodeName '/out'];
                hiddenOutputTensorName  = [lstmNodeName '/hidden'];
                cellOutputTensorName    = [lstmNodeName '/cell'];
                if hasSequenceOutput
                    lstmOutput      = {lstmOutputTensorName hiddenOutputTensorName cellOutputTensorName};
                else
                    lstmOutput      = {'' hiddenOutputTensorName cellOutputTensorName};
                end
            elseif hasSequenceOutput
                lstmOutputTensorName    = lstmNodeName;
                lstmOutput         = {lstmOutputTensorName};          % Output is Y=onnxName
            else
                lstmOutputTensorName    = lstmNodeName;
                lstmOutput         = {'', lstmOutputTensorName};      % Output is Y='', Y_h=onnxName
            end

            % Set attributes
            iofAct	= iTranslateActivationFunction(NNTLayer.GateActivationFunction, NNTLayer.Name);
            cellAct	= iTranslateActivationFunction(NNTLayer.StateActivationFunction, NNTLayer.Name);
            hidAct	= cellAct;

            % Make the lstm node
            [lstmNode, lstmInits] = createNodeProto(this, 'LSTMBidirectional', lstmNodeName, lstmInput, lstmOutput, ...
                NNTLayer.InputSize, hasSequenceOutput, ...
                iofAct, cellAct, hidAct, NNTLayer.InputWeights, NNTLayer.RecurrentWeights, NNTLayer.Bias, ...
                NNTLayer.NumHiddenUnits);

            % Create nodes to transform the LSTM's main output tensor.
            lstmMainOutputTensorName = lstmOutputTensorName;
            if NNTLayer.HasStateOutputs && ~hasSequenceOutput
                lstmMainOutputTensorName = hiddenOutputTensorName;
            end
            if hasSequenceOutput
                % Main output is Y. Do T2BC --> TBC
                [lstmMainOutputNodeProtos, lstmMainOutputInitializers] = createNodesToTransformT2BCtoTBC(...
                    this, lstmMainOutputTensorName, lstmNodeName, existingNodeNames);
            else
                % Main output is Yh. Do 2BC --> BC
                [lstmMainOutputNodeProtos, lstmMainOutputInitializers] = createNodesToTransform2BCtoBC(...
                    this, lstmMainOutputTensorName, '', lstmNodeName, existingNodeNames);
            end
            lstmOutputTensorName = lstmMainOutputNodeProtos(end).name;

            % Create nodes to transform the state outputs if necessary.
            if NNTLayer.HasStateOutputs
                [hiddenOutputNodeProtos, hiddenOutputInitializers] = createNodesToTransform2BCtoBC(...
                    this, hiddenOutputTensorName, 'Hidden_Output', lstmNodeName, existingNodeNames);
                [cellOutputNodeProtos, cellOutputInitializers] = createNodesToTransform2BCtoBC(...
                    this, cellOutputTensorName, 'Cell_Output', lstmNodeName, existingNodeNames);
                hiddenOutputTensorName = hiddenOutputNodeProtos(end).name;
                cellOutputTensorName = cellOutputNodeProtos(end).name;
            end

            % Collect the appropriate set of nodes and initializers
            % First, add the input processing nodes and initializers
            nodeProto = [nodeProto, unsqueezeNode, StateInputPreprocessingNodes];
            parameterInitializers = [unsqueezeInit, StateInputPreprocessingInits];

            % Second, add the LSTM node and its main output
            % processing nodes and initializers
            nodeProto = [nodeProto, lstmNode, lstmMainOutputNodeProtos];
            parameterInitializers = [parameterInitializers, lstmInits, lstmMainOutputInitializers];

            % Third, add the Yh and Yc output processing nodes and
            % initializers
            if NNTLayer.HasStateOutputs
                nodeProto = [nodeProto, hiddenOutputNodeProtos, cellOutputNodeProtos];
                parameterInitializers = [parameterInitializers, hiddenOutputInitializers, cellOutputInitializers];
            end

            % Determine layer output port names. These are the names used
            % by the MATLAB representation of the layer
            onnxNamePort = this.NNTLayer.Name;
            lstmOutputPort = this.NNTLayer.Name;
            if NNTLayer.HasStateOutputs
                lstmOutputPort = [this.NNTLayer.Name '/out'];
                hiddenOutputPort = [this.NNTLayer.Name '/hidden'];
                cellOutputPort = [this.NNTLayer.Name '/cell'];
            end

            % Update maps
            TensorNameMap(onnxNamePort) = lstmOutputTensorName;
            TensorNameMap(lstmOutputPort) = lstmOutputTensorName;
            TensorLayoutMap(lstmOutputTensorName) = lstmOutputTensorLayout;
            if NNTLayer.HasStateOutputs
                TensorNameMap(hiddenOutputPort) = hiddenOutputTensorName;
                TensorLayoutMap(hiddenOutputTensorName) = 'nc';
                TensorNameMap(cellOutputPort) = cellOutputTensorName;
                TensorLayoutMap(cellOutputTensorName) = 'nc';
            end

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end

        function [nodeProtos, initializers] = createNodesToTransformBCto2BC(this, tensorNameToTransform, ...
                newNodeNameStem, lstmNodeName, existingNodeNames, TensorNameMap, TensorLayoutMap)
            % Create a node sequence to transform BC to 2BC by splitting
            % the C dimension.
            import nnet.internal.cnn.onnx.*
            if ~isequal(TensorLayoutMap(tensorNameToTransform), 'nc')
                error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end

            % Reshape takes the shape as a tensor input. So we create a
            % TensorProto with an initializer.
            % Do BC --> B2C
            shapeTensor              = TensorProto;
            shapeTensor.name         = [lstmNodeName '_Reshape_' newNodeNameStem '_Shape'];
            shapeTensor.data_type    = TensorProto_DataType.INT64;
            shapeTensor.raw_data     = rawData(int64([0 2 -1]));
            shapeTensor.dims         = dimVector(3,1);

            reshapeName             = [lstmNodeName '_Reshape_' newNodeNameStem];
            reshapeName             = makeUniqueName([existingNodeNames, {lstmNodeName}], reshapeName);
            [reshapeNode, ~] = createNodeProto(this, 'SISO', 'Reshape', reshapeName, ...
                {tensorNameToTransform, shapeTensor.name}, {reshapeName});

            % Do B2C --> 2BC
            permuteName            = [lstmNodeName '_Permute_' newNodeNameStem];
            permuteName            = makeUniqueName([existingNodeNames, {lstmNodeName}], permuteName);
            [permuteNode, ~] = createNodeProto(this, 'Transpose', permuteName, ...
                {reshapeName}, {permuteName}, [1 0 2]);

            nodeProtos = [reshapeNode, permuteNode];
            initializers = shapeTensor;
        end

        function [nodeProtos, initializers] = createNodesToTransform2BCtoBC(this, tensorNameToTransform, ...
                newNodeNameStem, lstmNodeName, existingNodeNames)
            % Create a node sequence to transform 2BC to BC by reshaping
            % the '2' and 'C' dimensions into a single dimension.
            import nnet.internal.cnn.onnx.*
            statePerm  = [1 0 2];
            stateShape = [0 -1];

            % Transpose 2BC to B2C
            transposeName            = [lstmNodeName '_Transpose_' newNodeNameStem];
            transposeName            = makeUniqueName([existingNodeNames, {lstmNodeName}], transposeName);
            [transposeNode, ~] = createNodeProto(this, 'Transpose', transposeName, ...
                {tensorNameToTransform}, {transposeName}, statePerm);

            % Reshape B2C to BC. Reshape takes the shape as a tensor
            % input. So we create a TensorProto with an initializer.
            shapeTensor              = TensorProto;
            shapeTensor.name         = [lstmNodeName '_Reshape_' newNodeNameStem '_Shape'];
            shapeTensor.data_type    = TensorProto_DataType.INT64;
            shapeTensor.raw_data     = rawData(int64(stateShape));
            shapeTensor.dims         = dimVector(numel(stateShape),1);

            reshapeName             = [lstmNodeName '_Reshape_' newNodeNameStem];
            reshapeName             = makeUniqueName([existingNodeNames, {lstmNodeName, transposeName}], reshapeName);
            [reshapeNode, ~] = createNodeProto(this, 'SISO', 'Reshape', reshapeName, ...
                {transposeName, shapeTensor.name}, {reshapeName});

            nodeProtos = [transposeNode, reshapeNode];
            initializers = [shapeTensor];
        end

        function [nodeProtos, initializers] = createNodesToTransformT2BCtoTBC(this, tensorNameToTransform, ...
                lstmNodeName, existingNodeNames)
            % Create a node sequence to transform T2BC to TBC by reshaping
            % the '2' and 'C' dimensions into a single dimension.
            import nnet.internal.cnn.onnx.*
            perm  = [0 2 1 3];
            shape = [0 0 -1];

            transposeName            = [lstmNodeName '_Transpose'];
            transposeName            = makeUniqueName([existingNodeNames, {lstmNodeName}], transposeName);
            [transposeNode, ~] = createNodeProto(this, 'Transpose', transposeName, ...
                {tensorNameToTransform}, {transposeName}, perm);

            % Reshape takes the shape as a tensor input. So we create a
            % TensorProto with an initializer.
            reshapeTensor              = TensorProto;
            reshapeTensor.name         = [lstmNodeName '_Reshape_shape'];
            reshapeTensor.data_type    = TensorProto_DataType.INT64;
            reshapeTensor.raw_data     = rawData(int64(shape));
            reshapeTensor.dims         = dimVector(numel(shape),1);

            reshapeName             = [lstmNodeName '_Reshape'];
            reshapeName             = makeUniqueName([existingNodeNames, {lstmNodeName, transposeName}], reshapeName);
            [reshapeNode, ~] = createNodeProto(this, 'SISO', 'Reshape', reshapeName, ...
                {transposeName, reshapeTensor.name}, {reshapeName});

            nodeProtos = [transposeNode, reshapeNode];
            initializers = reshapeTensor;
        end

        function [nodes, inits, lstmInputNames] = createStateInputPreprocessingNodes(...
                this, NNTLayer, lstmNodeName, inputTensorNames, lstmInputNames, existingNodeNames, TensorNameMap, TensorLayoutMap)
            % Create nodes to transform the state inputs prior to LSTM
            % node. Update lstmInputNames with the names output by these
            % nodes.
            import nnet.internal.cnn.onnx.*

            if NNTLayer.HasStateInputs
                % There are state inputs.
                [hiddenInputNodeProtos, hiddenInputInitializers] = createNodesToTransformBCto2BC(...
                    this, inputTensorNames{2}, 'Input_Hidden', lstmNodeName, existingNodeNames, TensorNameMap, TensorLayoutMap);
                [cellInputNodeProtos, cellInputInitializers] = createNodesToTransformBCto2BC(...
                    this, inputTensorNames{3}, 'Input_Cell', lstmNodeName, existingNodeNames, TensorNameMap, TensorLayoutMap);
                nodes               = [hiddenInputNodeProtos, cellInputNodeProtos];
                inits               = [hiddenInputInitializers, cellInputInitializers];
                lstmInputNames{6}   = hiddenInputNodeProtos(end).name;
                lstmInputNames{7}   = cellInputNodeProtos(end).name;
            else
                % There are no state inputs. Insert nodes that gather the
                % batchSize from X and expand the state initializers to
                % their required shape using Tile. We want to create 2BH in
                % onnx, given 2Hx1 matlab data and the ONNX X input of shape
                % TBC.
                %
                % First we must take the 2Hx1 MATLAB data and turn it into
                % [2,1,H] ONNX data.
                H = NNTLayer.NumHiddenUnits;
                initial_h = permute(reshape(NNTLayer.HiddenState, [H, 2, 1]), [2 3 1]);    % NNT size is [2*H, 1]. ONNX size is [2, 1, H]
                initial_c = permute(reshape(NNTLayer.CellState, [H, 2, 1]), [2 3 1]);      % NNT size is [2*H, 1]. ONNX size is [2, 1, H]

                % The following ONNX code has been carefully chosen so
                % it works for all opsets>=6:
                %
                % S = Shape(X)					                            S is rank 1, int64  = [T B C].
                % B = Gather(S, init([1]), axis=0)			                Make indices [1] (rank 1) to get rank 1 output.
                % repeats = Concat(init([1]), B, init([1]), axis=0).	    Build the vector [1,B,1]
                % newH = Tile(H, repeats)                                   Turn shape [2,1,H] into [2,B,H]
                % newC = Tile(C, repeats)                                   Turn shape [2,1,H] into [2,B,H]

                % Shape(X) node:
                sNodeName = [lstmNodeName '_Shape1'];
                [sNode, ~]  = createNodeProto(this, 'SISO', 'Shape', sNodeName, lstmInputNames(1), {sNodeName});

                % Gather node:
                gNodeName = [lstmNodeName '_Gather1'];
                gIndicesName = [lstmNodeName '_Gather1Indices'];
                gInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(gIndicesName, [1], 1, TensorProto_DataType.INT64);
                [gNode, ~]  = createNodeProto(this, 'Gather', gNodeName, {sNodeName, gIndicesName}, {gNodeName}, 0);

                % Concat node:
                cNodeName = [lstmNodeName '_Concat1'];
                cConst1Name = [lstmNodeName '_ConcatConst1'];
                cConst1Init = nnet.internal.cnn.onnx.makeTensorProtoOfType(cConst1Name, [1], 1, TensorProto_DataType.INT64);
                [cNode, ~]  = createNodeProto(this, 'Concat', cNodeName, {cConst1Name, gNodeName, cConst1Name}, {cNodeName}, 0);

                % Tile(H) node:
                tNodeName1 = [lstmNodeName '_TileH'];
                [tNode1, ~]  = createNodeProto(this, 'SISO', 'Tile', tNodeName1, {lstmInputNames{6}, cNodeName}, {tNodeName1});

                % Tile(C) node:
                tNodeName2 = [lstmNodeName '_TileC'];
                [tNode2, ~]  = createNodeProto(this, 'SISO', 'Tile', tNodeName2, {lstmInputNames{7}, cNodeName}, {tNodeName2});

                % Create initializers for the raw initial_h and initial_c
                % input tensors, of shape [1,1,nH].
                HInit       = makeTensorProtoOfType(lstmInputNames{6}, [2,1,NNTLayer.NumHiddenUnits], initial_h, TensorProto_DataType.FLOAT);
                CInit       = makeTensorProtoOfType(lstmInputNames{7}, [2,1,NNTLayer.NumHiddenUnits], initial_c, TensorProto_DataType.FLOAT);

                % Return the nodes and inits and update the LSTM input names
                nodes               = [sNode, gNode, cNode, tNode1, tNode2];
                inits               = [gInit, cConst1Init, HInit, CInit];
                lstmInputNames{6}   = tNodeName1;
                lstmInputNames{7}   = tNodeName2;
            end
        end

    end
end


function s = iTranslateActivationFunction(s, layerName)
switch s
    case 'sigmoid'
        s = 'Sigmoid';
    case 'softsign'
        s = 'Softsign';
    case 'tanh'
        s = 'Tanh';
    case 'relu'
        s = 'Relu';
    case 'hard-sigmoid'
        s = 'HardSigmoid';
    otherwise
        error(message('nnet_cnn_onnx:onnx:InvalidActivationFunction', s, layerName));
end
end
