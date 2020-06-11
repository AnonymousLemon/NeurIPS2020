function [A_train, b_train, A_test, b_test] = loadData(dataName)

fprintf('\n..................LOADING DATA.................\n');

assert(strcmp(dataName,'arcene') || ... % start of classification names
    strcmp(dataName,'dorothea') || ...
    strcmp(dataName,'20News') || ...
    strcmp(dataName,'covetype') || ...
    strcmp(dataName, 'mnist') || ... 
    strcmp(dataName, 'UJIIndoorLoc-classification') || ... 
    strcmp(dataName, 'gisette') || ...
    strcmp(dataName, 'hapt') || ...
    strcmp(dataName, 'cifar10') || ...
    strcmp(dataName, 'drive-diagnostics') || ... % end of classification names
    strcmp(dataName, 'blogdata') ||... % start of regression names
    strcmp(dataName, 'power-plant') || ...
    strcmp(dataName, 'news-populairty') || ...
    strcmp(dataName, 'housing') || ...
    strcmp(dataName, 'blog-feedback') || ...
    strcmp(dataName, 'forest-fire') || ...
    strcmp(dataName, 'UJIIndoorLoc-regression')); % end of regression names


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% BEGIN CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% Beging Arcene: Binary Classificaion  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'arcene')
    A_train = importdata('../../Data/arcene/arcene_train.data');
    labels_train_raw = importdata('../../Data/arcene/arcene_train.labels');
    b_train = getClassLabels(labels_train_raw, true);
    A_test = importdata('../../Data/arcene/arcene_valid.data');
    labels_test_raw= importdata('../../Data/arcene/arcene_valid.labels');
    b_test = getClassLabels(labels_test_raw, false);
end
%%%%%%%%%%%%% End Arcene %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% Begin Dorothea: Binary Classificaion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'dorothea')
    load('../../Data/dorothea/dorothea_train.mat');
    A_train = doroth_train_data;
    b_train = getClassLabels(doroth_train_labels, true);
    load('../../Data/dorothea/dorothea_valid.mat');
    A_test = doroth_valid_data;
    b_test = getClassLabels(doroth_valid_labels, false);
end
%%%%%%%%% End Dorothea %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% Beging 20news: Multi-Class Classificaion  %%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, '20News')
    %     A_raw = load('../../Data/news/train.data');
    %     A_raw = sparse(A_raw(:,1),A_raw(:,2),A_raw(:,3));
    %     labels_raw = load('../../Data/news/train.label');
    %     b_raw = getClassLabels(labels_raw, false);
    %     test_index = randsample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    %     train_index = setdiff((1:size(A_raw,1))',test_index);
    %     assert(isempty(intersect(test_index,train_index)));
    %     A_train = A_raw(train_index,:);
    %     b_train = b_raw(train_index,:);
    %     b_train = b_train(:,1:end-1);
    %     A_test = A_raw(test_index,:);
    %     b_test = b_raw(test_index,:);
%     A_train = importdata('../../Data/news/sudhir/train_mat.txt');
%     A_train = sparse(A_train(:,1),A_train(:,2),A_train(:,3));
%     labels_train_raw = load('../../Data/news/sudhir/train_vec.txt');
%     b_train = getClassLabels(labels_train_raw, true);
%     A_test = load('../../Data/news/sudhir/test_mat.txt');
%     A_test = sparse(A_test(:,1),A_test(:,2),A_test(:,3));
%     labels_test_raw = load('../../Data/news/sudhir/test_vec.txt');
%     b_test = getClassLabels(labels_test_raw , false);
    A_train = load('../../Data/news/sudhir/train_mat.txt');
    labels_train_raw = load('../../Data/news/sudhir/train_vec.txt');    
    b_train = getClassLabels(labels_train_raw, true);

    A_test = load('../../Data/news/sudhir/test_mat.txt');
    labels_test_raw = load('../../Data/news/sudhir/test_vec.txt');
    b_test = getClassLabels(labels_test_raw , false);

    max_col = max( max(A_train(:, 2)), max(A_test(:, 2) ));

    A_train = sparse(A_train(:,1),A_train(:,2),A_train(:,3), max(A_train(:, 1)), max_col);
    A_test = sparse( A_test( :, 1), A_test(:, 2), A_test(:, 3), max(A_test(:, 1)), max_col);

end
%%%%%%%%% End 20news %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% Begin CoveType: Multi-Class Classificaion %%%%%%%%%%%%%
if strcmp(dataName, 'covetype')
    DD_train = importdata('../../Data/CoveType/covtype.data');
    A_raw = DD_train( :, 1:54 );
    labels_raw = DD_train( :, 55 );
    b_raw = getClassLabels(labels_raw, false);
    test_index = randsample(size(A_raw,1), ceil(0.25*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    b_train = b_train(:,1:end-1);
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end
%%%%%%%%%%%%%% End CoveType %%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'mnist')
    A_train = loadMNISTImages('../../Data/MNIST/train-images-idx3-ubyte');
    A_train = A_train';
    labels_train_raw = loadMNISTLabels('../../Data/MNIST/train-labels-idx1-ubyte');
    b_train = getClassLabels(labels_train_raw, true);
    A_test = loadMNISTImages('../../Data/MNIST/t10k-images-idx3-ubyte');
    A_test = A_test';
    labels_test_raw = loadMNISTLabels('../../Data/MNIST/t10k-labels-idx1-ubyte');
    b_test = getClassLabels(labels_test_raw, false);    
end

% %%%%%%%%% UJIIndoorLoc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% http://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc
% Attribute 523 (Floor)
% Attribute 524 (BuildingID)
if strcmp(dataName, 'UJIIndoorLoc-classification')
    attr = 523;
    DD_train = xlsread('../../Data/UJIndoorLoc/trainingData.csv');
    A_train = DD_train( :, 1:520 );
    labels_train_raw = DD_train(:, attr);
    b_train = getClassLabels(labels_train_raw, true);
    DD_test = xlsread('../../Data/UJIndoorLoc/validationData.csv');
    A_test = DD_test( : , 1:520 );
    labels_test_raw = DD_test( : , attr );
    b_test = getClassLabels(labels_test_raw, false);
end

%%%%%%%%% Beging gisette: Multi-Class Classificaion  %%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'gisette')
    A_train = load('../../Data/gisette/gisette_train.data');
    labels_train_raw = load('../../Data/gisette/gisette_train.labels');
    b_train = getClassLabels(labels_train_raw, true);
    A_test = load('../../Data/gisette/gisette_valid.data');
    labels_test_raw = load('../../Data/gisette/gisette_valid.labels');
    b_test = getClassLabels(labels_test_raw , false);
end

%%%%%%%%% Beging gisette: Multi-Class Classificaion  %%%%%%%%%%%%%%%%%%%%%%
%% http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
if strcmp(dataName, 'hapt')
    A_train = importdata('../../Data/HAPT/Train/X_train.txt');
%     A_train = normalizeData(A_train);
    labels_train_raw = load('../../Data/HAPT/Train/y_train.txt');
    b_train = getClassLabels(labels_train_raw, true);
    A_test = load('../../Data/HAPT/Test/X_test.txt');
%     A_test = normalizeData(A_test);
    labels_test_raw = load('../../Data/HAPT/Test/y_test.txt');
    b_test = getClassLabels(labels_test_raw , false);
end

if  strcmp(dataName, 'cifar10')
    DD_train_1 = importdata('../../Data/cifar10/data_batch_1.mat');
    DD_train_2 = importdata('../../Data/cifar10/data_batch_2.mat');
    DD_train_3 = importdata('../../Data/cifar10/data_batch_3.mat');
    DD_train_4 = importdata('../../Data/cifar10/data_batch_4.mat');
    DD_train_5 = importdata('../../Data/cifar10/data_batch_5.mat');
    A_train_0 = double([DD_train_1.data; DD_train_2.data; DD_train_3.data; DD_train_4.data; DD_train_5.data]);
    labels_train_raw = [DD_train_1.labels; DD_train_2.labels; DD_train_3.labels; DD_train_4.labels; DD_train_5.labels];
    b_train = getClassLabels(labels_train_raw, true);
    DD_test = importdata('../../Data/cifar10/test_batch.mat');
    A_test_0 = double(DD_test.data);
    labels_test_raw = DD_test.labels;
    b_test = getClassLabels(labels_test_raw, false);
    
    A_train = A_train_0;
    A_test  = A_test_0; 
%     % Tranform to wavelet basis to make sparse
%     img_size = 32;
%     cut_off  = 50;
%     max_level = log2(img_size);
%     level = max_level;
%     jmin = max_level - level;
%     A_train = zeros(size(A_train_0,1),size(A_train_0,2));
%     for i = 1: size(A_train,1)
%         for j = 1:3
%             img = A_train_0(i,img_size*img_size*(j-1) + 1:img_size*img_size*j);
%             img = reshape(img, img_size,img_size);
%             %figure(); imagesc(img);
%             w0 = perform_wavortho_transf(img, jmin, +1);
%             w = w0;
%             w(abs(w) <= cut_off) = 0;
%             %img2 = perform_wavortho_transf(w, jmin, -1);
%             %figure(); imagesc(img2);
%             fprintf('i: %g, j: %g, Sparsification Ration: %g\n',i,j,nnz(w)/nnz(w0));
%             A_train(i,img_size*img_size*(j-1) + 1:img_size*img_size*j) = w(:);
%         end
%     end
%     A_test = zeros(size(A_test_0,1),size(A_test_0,2));
%     for i = 1: size(A_test,1)
%         for j = 1:3
%             img = A_test_0(i,img_size*img_size*(j-1) + 1:img_size*img_size*j);
%             img = reshape(img, img_size,img_size);
%             %figure(); imagesc(img);
%             w0 = perform_wavortho_transf(img, jmin, +1);
%             w = w0;
%             w(abs(w) <= cut_off) = 0;
%             %img2 = perform_wavortho_transf(w, jmin, -1);
%             %figure(); imagesc(img2);
%             fprintf('i: %g, j: %g, Sparsification Ration: %g\n',i,j,nnz(w)/nnz(w0));
%             A_test(i,img_size*img_size*(j-1) + 1:img_size*img_size*j) = w(:);
%         end
%     end
% A_train = sparse(A_train);
% A_test = sparse(A_test);
end

if strcmp(dataName, 'drive-diagnostics')
    %     A_train = importdata('../../Data/drive-diagnostics/diagnosis_train_mat.txt');
    %     labels_train_raw = load('../../Data/drive-diagnostics/diagnosis_train_vec.txt');
    %     b_train = getClassLabels(labels_train_raw, true);
    %     A_test = load('../../Data/drive-diagnostics/diagnosis_test_mat.txt');
    %     labels_test_raw = load('../../Data/drive-diagnostics/diagnosis_test_vec.txt');
    %     b_test = getClassLabels(labels_test_raw , false);
    A_train = importdata('../../Data/drive-diagnostics/train_mat.txt');
    labels_train_raw = load('../../Data/drive-diagnostics/train_vec.txt');
    b_train = getClassLabels(labels_train_raw, true);
    A_test = load('../../Data/drive-diagnostics/test_mat.txt');
    labels_test_raw = load('../../Data/drive-diagnostics/test_vec.txt');
    b_test = getClassLabels(labels_test_raw , false);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% END CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% BEGIN RERGRESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% Begin blog data: Regression %%%%%%%%%%%%%
if strcmp(dataName, 'blogdata')
    DD_train = csvread('../../Data/sample_data/blogData_test-2012.03.20.00_00.csv');
    A_raw = DD_train( :, 1:50 );
    b_raw = DD_train( :, 51 );
    test_index = randsample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% Power Plant %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'power-plant')
    DD_train = csvread('../../Data/CCPP/power_plant.csv');
    A_raw = DD_train( :, 1:4 );
    b_raw = DD_train(:, 5);
    test_index = randsample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% News Popularity %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'news-populairty')
    DD_train = csvread('../../Data/OnlineNewsPopularity/OnlineNewsPopularity_no_url.csv');
    A_raw = DD_train( :, 1:59 );
    b_raw = DD_train(:, 60);
    test_index = randsample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% Housing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'housing')
    DD_train = csvread('../../Data/housing/housing.csv');
    A_raw = DD_train( :, 1:13 );
    b_raw = DD_train(:, 14);
    test_index = randsample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% Blog Feedback %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'blog-feedback')
    DD_train = csvread('../../Data/BlogFeedback/blogData_train.csv');
    A_raw = DD_train( :, 1:280 );
    b_raw = DD_train(:, 281);
    test_index = randsample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% Forest Fire %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataName, 'forest-fire')
    DD_train = csvread('../../Data/forest_fire/forestfires.csv');
    A_raw = DD_train( :, 1:10 );
    b_raw = DD_train(:, 11);
    test_index = randsample(size(A_raw,1), ceil(0.1*size(A_raw,1)));
    train_index = setdiff((1:size(A_raw,1))',test_index);
    assert(isempty(intersect(test_index,train_index)));
    A_train = A_raw( train_index, : );
    b_train = b_raw( train_index, : );
    A_test = A_raw( test_index, : );
    b_test = b_raw( test_index, :);
end

% %%%%%%%%% UJIIndoorLoc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% http://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc
% Attribute 521 (Longitude): Longitude. Negative real values from -7695.9387549299299000 to -7299.786516730871000 
% Attribute 522 (Latitude): Latitude. Positive real values from 4864745.7450159714 to 4865017.3646842018. 
if strcmp(dataName, 'UJIIndoorLoc-regression')
    DD_train = xlsread('../../Data/UJIndoorLoc/trainingData.csv');
    A_train = DD_train( :, 1:520 );
    b_train = DD_train(:, 521);
    DD_test = xlsread('../../Data/UJIndoorLoc/validationData.csv');
    A_test = DD_test( : , 1:520 );
    b_test = DD_test( : , 521 );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% END RERGRESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('..................DONE LOADING DATA.................\n');
end

%%%%%%%%%% CT Slice %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DD = importdata('../../Data/CTSlice/slice_localization_data.csv');
% A = DD.data(:,2:385);
% b = 0.5*(1+sign(randn(size(A,1),1)));
% ubbb = sort(unique(b));
% assert(ubbb(1) == 0 && ubbb(2) == 1);

%%%%%%%%%% Parkinson %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% D = importdata('../../Data/Parkinson/train_data.txt');
% A = D(:,2:27); b = D(:,29);
% ubbb = sort(unique(b));
% assert(ubbb(1) == 0 && ubbb(2) == 1);
