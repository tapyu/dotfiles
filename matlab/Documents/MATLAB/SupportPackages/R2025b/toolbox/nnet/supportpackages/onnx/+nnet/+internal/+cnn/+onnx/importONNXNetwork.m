function Network = importONNXNetwork(filename, NameValueArgs)

%   Copyright 2018-2024 The MathWorks, Inc.

arguments 
    filename {mustBeTextScalar}
    NameValueArgs.ClassNames = {}
    NameValueArgs.Classes = 'auto'
    NameValueArgs.PackageName {mustBeTextScalar} = ''
    NameValueArgs.Namespace {mustBeTextScalar} = ''
    NameValueArgs.OutputLayerType {mustBeTextScalar, mustBeMember(NameValueArgs.OutputLayerType, {'', 'classification', 'regression', 'pixelclassification'})} = ''
    NameValueArgs.ImageInputSize (1, :) {nnet.internal.cnn.onnx.util.validateIONImageInputSize(NameValueArgs.ImageInputSize)} = []
    NameValueArgs.TargetNetwork {mustBeTextScalar, mustBeMember(NameValueArgs.TargetNetwork, {'dlnetwork', 'dagnetwork'})} = 'dagnetwork'
    NameValueArgs.GenerateCustomLayers (1,1) logical = true
    NameValueArgs.InputDataFormats = ''
    NameValueArgs.OutputDataFormats = ''
    NameValueArgs.FoldConstants {mustBeTextScalar, mustBeMember(NameValueArgs.FoldConstants, {'deep', 'shallow', 'none'})} = 'deep'
    NameValueArgs.IsInputForwardONNX (1,1) logical = false
    NameValueArgs.CustomLayerContent {mustBeTextScalar, mustBeMember(NameValueArgs.CustomLayerContent, {'groupedOperators', 'individualOperators', 'entireNetwork'})} = 'groupedOperators'
    NameValueArgs.GenerateCoderFiles (1,1) logical = false
end 

% Warn about importONNXNetwork deprecation
if ~NameValueArgs.IsInputForwardONNX
    iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:WarnAPIDeprecation', 'importONNXNetwork');
end

% Warn about PackageName argument deprecation
if(~isempty(NameValueArgs.PackageName))
    iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:WarnArgumentDeprecation');
end
% Copy Namespace value to PackageName if Namespace is not empty
if(~isempty(NameValueArgs.Namespace))
    NameValueArgs.PackageName = NameValueArgs.Namespace;
end

nnet.internal.cnn.onnx.setAdditionalResourceLocation();     % For SPKG resource catalog.
[Filename, OutputLayerType, userInputFormats, userOutputFormats, ...
    GenerateCustomLayers, Classes, TargetNetwork, UserImageInputSize, ...
    PackageName, OrigPackageName, FoldConstants, IsInputForwardONNX,...
    CustomLayerContent, GenerateCoderFiles] = iValidateInputs(filename, NameValueArgs);

% If 'Classes' are auto, and the user passed
% OutputLayerType=classification, warn the user
iWarnIfClassesAuto(Classes, OutputLayerType);

% Import
modelProto = nnet.internal.cnn.onnx.ModelProto(Filename);
nnet.internal.cnn.onnx.warnIfFutureIR(modelProto);
iCheckEmptyModel(modelProto, filename);
iCheckExternalData(modelProto, filename);
modelTranslationIntoLayers = nnet.internal.cnn.onnx.ModelTranslationIntoLayers(modelProto, ...
    OutputLayerType, userInputFormats, userOutputFormats, TargetNetwork, GenerateCustomLayers, ...
    UserImageInputSize, Classes, PackageName, OrigPackageName, FoldConstants, IsInputForwardONNX,...
    CustomLayerContent, GenerateCoderFiles);
lg = modelTranslationIntoLayers.LayerGraph;
translationIssues = modelTranslationIntoLayers.TranslationIssues;
iErrorIfAnyUnlicensedOps(translationIssues);
[issueCounts, issueStrings] = iParseTranslationIssues(translationIssues);
hasTranslationIssues = false;
if numel(issueCounts) > 0
    iHandleTranslationIssues(issueCounts, issueStrings, IsInputForwardONNX);
    hasTranslationIssues = true;
end
C = iSuppressAutoClassesWarning(); %#ok<NASGU>
if isequal(modelTranslationIntoLayers.TargetNetwork, 'dagnetwork')
    Network = assembleDAGNetwork(lg);
else
    Network = constructDlnetwork(lg, modelTranslationIntoLayers.ModelOutputNames, hasTranslationIssues);
end
end

function Network = assembleDAGNetwork(lg)
try
    Network = assembleNetwork(lg);
catch me
    msg = join([
        string(message('nnet_cnn_onnx:onnx:CantAssembleDAGNetwork').getString())
        newline
        string(message('nnet_cnn_onnx:onnx:UseTargetNetworkDlnetwork').getString())
        newline
        string(message('nnet_cnn_onnx:onnx:UseImportONNXLayers').getString())
        newline
        string(message('nnet_cnn_onnx:onnx:UseImportONNXFunction').getString())
        ], newline);
    summary = message('nnet_cnn_onnx:onnx:GenericMessage', msg);
    throwAsCaller(MException(summary));
end
end

function Network = constructDlnetwork(lg, modelOutputs, hasTranslationIssues)
Network = dlnetwork(lg, 'Initialize', false);
% Remove any extraneous outputs
if ~isempty(modelOutputs)
    Network.OutputNames = strcat(modelOutputs,'Output');
end
hasCustomInputLayer = false;
for i = 1:numel(lg.Layers)
    if isa(lg.Layers(i),'nnet.onnx.layer.CustomInputLayer')||isa(lg.Layers(i),'nnet.onnx.layer.CustomInputLayerMultiOutput')
        hasCustomInputLayer = true;
        break;    
    end
end
if hasTranslationIssues
    % If there are any translation issues, return uninitialized dlnetwork
    iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:INFOTranslationIssues');    
elseif hasCustomInputLayer
    % If there are any custom input layers, return uninitialized dlnetwork
    iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:UninitializedDlnet');
else
    try
        % Try to initialize the dlnetwork
        Network = expandLayers(Network);
        Network = initialize(Network);
    catch ME 
        % Return the uninitialized dlnetwork and inform the user
        if strcmp(ME.identifier, 'nnet_cnn:internal:cnn:analyzer:NetworkAnalyzer:NetworkHasErrors') && ~isempty(ME.cause)
            if strcmp(ME.cause{1}.identifier,'nnet_cnn:cnn:layer:CustomLayer:PredictErrored') && ~isempty(ME.cause{1}.cause)...
                && strcmp(ME.cause{1}.cause{1}.identifier, 'nnet_cnn_onnx:onnx:WrongFixedBatchSize')
                % Unable to initialize because of one or more nnet.onnx.layer.VerifyBatchSizeLayer
                summary = '';
                for i = 1:numel(ME.cause)
                    if strcmp(ME.cause{i}.identifier,'nnet_cnn:cnn:layer:CustomLayer:PredictErrored') && ~isempty(ME.cause{i}.cause)...
                       && strcmp(ME.cause{i}.cause{1}.identifier, 'nnet_cnn_onnx:onnx:WrongFixedBatchSize')
                        % Predict issue in nnet.onnx.layer.VerifyBatchSizeLayer predict call
                        % during dlnetwork initialization
                        innerException = ME.cause{i}.cause{1};      
                        exceptionMessage = strsplit(innerException.message,newline);
                        summary = [summary '     ' exceptionMessage{end} newline]; %#ok<AGROW>
                    end
                end 
                iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:InitializationFailedDueToKnownIssue', summary);
            elseif strcmp(ME.cause{1}.identifier,'nnet_cnn:cnn:layer:CustomLayer:PredictErrored') && ~isempty(ME.cause{1}.cause)
                % Unable to initialize because of some other 
                % custom layer predict error
                summary = '';
                for i = 1:numel(ME.cause)
                    if strcmp(ME.cause{i}.identifier,'nnet_cnn:cnn:layer:CustomLayer:PredictErrored') && ~isempty(ME.cause{i}.cause)
                        % Prediction issue in a custom layer predict call
                        % during dlnetwork initialization
                        exceptionMessage = ME.cause{i}.cause{1}.message;
                        summary = [summary '     ' exceptionMessage]; %#ok<AGROW>
                    end
                end 
                iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:InitializationFailedDueToKnownIssue', summary);
            elseif strcmp(ME.cause{1}.identifier,'nnet_cnn:cnn:layer:CustomLayer:InitializeErrored') && ~isempty(ME.cause{1}.cause)
                % Unable to initialize because of custom layer
                % initialization error
                summary = '';
                for i = 1:numel(ME.cause)
                    if strcmp(ME.cause{i}.identifier,'nnet_cnn:cnn:layer:CustomLayer:InitializeErrored') && ~isempty(ME.cause{i}.cause)
                        % Initialization issue in a custom layer predict call
                        exceptionMessage = ME.cause{i}.cause{1}.message;
                        summary = [summary '     ' exceptionMessage]; %#ok<AGROW>
                    end
                end 
                iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:InitializationFailedDueToKnownIssue', summary);
            else
                % Some other cause, show the generic warning
                iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:DlnetInitializeFailed');
            end
        end
    end
end
end

function iHandleTranslationIssues(issueCounts, issueStrings, isInputForwardONNX)
if isInputForwardONNX
    % Warn about placeholder layers/functions for the new API and give
    % out an uninitialized dlnetwork
    iWarnIfContainsPlaceholders(issueCounts, issueStrings);
else
    % Error and suggest importONNXLayers/Function if call from
    % importONNXNetwork
    iErrorIfContainsPlaceholders(issueCounts, issueStrings);
end
end

function iErrorIfAnyUnlicensedOps(translationIssues)
% Look for translation issues 'nnet_cnn_onnx:onnx:noIPTForOperator'
% or 'nnet_cnn_onnx:onnx:noCVTForOperator' 
messages = arrayfun(@(issue) string(issue.Message.Identifier), translationIssues);
isUnlicensed = arrayfun(@(message) ismember(message, ["nnet_cnn_onnx:onnx:noIPTForOperator", "nnet_cnn_onnx:onnx:noCVTForOperator"]), messages);
if any(isUnlicensed)
    unlicensedIssues = translationIssues(isUnlicensed);
    firstUnlicensedIssue = unlicensedIssues(1);
    throwAsCaller(MException(firstUnlicensedIssue.Message));
end
end

function [IssueCounts, IssueStrings] = iParseTranslationIssues(translationIssues)
% Summarizes the issues encountered during translation
% Only report on issues that led to placeholder creation
translationIssues   = translationIssues(arrayfun(@(x) isa(x, 'nnet.internal.cnn.onnx.NodeTranslationError'), translationIssues));
errorMessages       = arrayfun(@(x) getMessageString(x), translationIssues, 'UniformOutput', false);
IssueStrings        = string(unique(errorMessages));

IssueCounts = zeros(numel(IssueStrings), 1);
for i=1:numel(IssueStrings)
    IssueCounts(i) = sum(strcmp(IssueStrings(i), errorMessages));
end
end

function iErrorIfContainsPlaceholders(issueCounts, issueStrings)
% Error if any unsupported layers, refer user to importONNXLayers
if numel(issueCounts) > 0
    prologue = message('nnet_cnn_onnx:onnx:CantBuildNetWithUnsupportedLayers').getString();
    unsupportedLayersString = "";
    for i=1:numel(issueCounts)
        unsupportedLayersString = join([unsupportedLayersString,...
            sprintf('%d operator(s)\t:\t%s', issueCounts(i), issueStrings(i))], newline);
    end
    epilogue = join([message('nnet_cnn_onnx:onnx:UseTargetNetworkDlnetwork').getString(),...
        newline, newline,...
        message('nnet_cnn_onnx:onnx:UseImportONNXLayers').getString(),...
        newline, newline,...
        message('nnet_cnn_onnx:onnx:UseImportONNXFunction').getString()], newline);

    summary = message('nnet_cnn_onnx:onnx:IssuesDuringImport', prologue, unsupportedLayersString, epilogue);
    throwAsCaller(MException(summary));
end
end

function iWarnIfContainsPlaceholders(placeholderIssueCounts, placeholderIssueStrings)
% Warn if any unsupported layers, refer user to importONNXFunction
prologue = "";
epilogue = message('nnet_cnn_onnx:onnx:UseImportONNXFunction').getString();
if numel(placeholderIssueCounts) > 0
    summary = iGetIssueSummary(placeholderIssueStrings, placeholderIssueCounts, 'nnet_cnn_onnx:onnx:UnsupportedLayerWarning'); 
    iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:IssuesDuringImport', prologue, summary, epilogue);
end
end

function issueSummary = iGetIssueSummary(issueStrings, issueCounts, issueID)
    issueSummary = [message(issueID).getString(), newline];
    for i=1:numel(issueCounts)
        issueSummary = join([issueSummary,...
            sprintf("%d operator(s)\t:\t%s", issueCounts(i), issueStrings(i))], newline);
    end
end

function [Filename, OutputLayerType, userInputFormats, userOutputFormats, ...
    GenerateCustomLayers, Classes, TargetNetwork, ImageInputSize, PackageName, OrigPackageName, FoldConstants, IsInputForwardONNX,...
    CustomLayerContent, GenerateCoderFiles] = iValidateInputs(filename, NameValueArgs)
Classes                 = iValidateClassesAndClassNames(NameValueArgs.Classes, NameValueArgs.ClassNames);
Filename                = iValidateFile(filename);
ImageInputSize          = double(NameValueArgs.ImageInputSize(:))';
OutputLayerType         = iValidateOutputLayerType(NameValueArgs.OutputLayerType);
userInputFormats        = nnet.internal.cnn.onnx.util.validateIONDataFormats(NameValueArgs.InputDataFormats, "InputDataFormats");
userOutputFormats       = nnet.internal.cnn.onnx.util.validateIONDataFormats(NameValueArgs.OutputDataFormats, "OutputDataFormats");
[PackageName, OrigPackageName] = iValidatePackageName(NameValueArgs.PackageName, filename);
GenerateCustomLayers    = NameValueArgs.GenerateCustomLayers;
TargetNetwork           = NameValueArgs.TargetNetwork;
FoldConstants           = NameValueArgs.FoldConstants;
IsInputForwardONNX      = NameValueArgs.IsInputForwardONNX;
CustomLayerContent         = NameValueArgs.CustomLayerContent;
GenerateCoderFiles         = NameValueArgs.GenerateCoderFiles;
end

function Classes = iValidateClassesAndClassNames(Classes, ClassNames)
    defaultClassNames = {};
    defaultClasses = 'auto';
    iAssertValidClasses(Classes);
    if iIsSpecified(ClassNames, defaultClassNames) && ...
            iIsSpecified(Classes, defaultClasses)
        throwAsCaller(MException(message(...
            'nnet_cnn_onnx:onnx:ClassesAndClassNamesNVP')))
    elseif iIsSpecified(ClassNames, defaultClassNames)
        iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:ClassNamesDeprecated'); 
        ClassNames = iValidateClassNames(ClassNames);
        Classes = categorical(ClassNames, ClassNames);
    elseif iIsSpecified(Classes, defaultClasses)
        Classes = iConvertClassesToCanonicalForm(Classes); 
    else
        % Not specified ClassNames nor Classes. Do nothing.
    end
end

function [packageName, OrigPackageName] = iValidatePackageName(OrigPackageName, onnxFilename)
if isempty(OrigPackageName)
    [~,OrigPackageName,~] = fileparts(onnxFilename);
end
if isvarname(OrigPackageName)
    packageName = OrigPackageName;
else
    packageName = matlab.lang.makeValidName(OrigPackageName);
end
end

function iCheckEmptyModel(modelProto, filename)
% Check if a layer graph is present in the imported model
if isempty(modelProto.graph)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:EmptyModel',filename)));
end
end

function iCheckExternalData(modelProto, filename)
% Check if External Data is used
if any(arrayfun(@(x) isequal(x.data_location,"EXTERNAL"), modelProto.graph.initializer))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:ExternalData',filename)));
end
end

function  ClassNames = iValidateClassNames(ClassNames)
if isstring(ClassNames)
    ClassNames = cellstr(ClassNames);
elseif ~iscellstr(ClassNames)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:InvalidClassNames')));
end
if ~isvector(ClassNames)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:InvalidClassNames')));
end
% make sure it's a column vector
ClassNames = ClassNames(:);
end

function iAssertValidClasses(value)
nnet.internal.cnn.layer.paramvalidation.validateClasses(value);
end

function classes = iConvertClassesToCanonicalForm(classes)
classes = ...
    nnet.internal.cnn.layer.paramvalidation.convertClassesToCanonicalForm(classes);
end

function OutputLayerType = iValidateOutputLayerType(OutputLayerType)
if ~isempty(OutputLayerType) 
    if isequal(OutputLayerType,'pixelclassification') && ~nnet.internal.cnn.onnx.isInstalledCVST
        throwAsCaller(MException(message('nnet_cnn_onnx:onnx:noCVSTForPixelClassification')));
    end
end
end

function Filepath = iValidateFile(Filename)
if ~(isa(Filename,'char') || isa(Filename,'string'))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:FirstArgString')));
end
% Check if the file extension is valid (i.e., if the filename ends with '.onnx')
[~,~,ext] = fileparts(Filename);
if (~strcmp(char(ext),'.onnx'))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:InvalidFileExtension', Filename)));
end
Filepath = which(char(Filename));
if isempty(Filepath) && exist(Filename, 'file')
    Filepath = char(Filename);
end
if ~exist(Filepath, 'file')
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:FileNotFound', Filename)));
end
end

function tf = iIsSpecified(value, defaultValue)
tf = ~isequal(convertStringsToChars(value), defaultValue);
end

function C = iSuppressAutoClassesWarning()
warnState = warning('off', 'nnet_cnn:internal:cnn:analyzer:NetworkAnalyzer:NetworkHasWarnings');
C = onCleanup(@()warning(warnState));
end

function tf = iIsAuto(val)
tf = isequal(string(val), "auto");
end

function tf = iIsClassification(val)
tf = isequal(string(val), "classification");
end

function iWarnIfClassesAuto(Classes, OutputLayerType)
if iIsAuto(Classes) && iIsClassification(OutputLayerType)
    % 'auto' classes passed. assembleNetwork will set them later.
    iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:FillingInClassNames');
end
end

function iWarningWithoutBacktrace(msgID, varargin)
backtrace = warning('query','backtrace');
warning('off','backtrace');
warning(message(msgID, varargin{:}));
warning(backtrace.state,'backtrace');
end