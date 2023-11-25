clc
clear all
close all


%% Problem definition



n = 1000;

% Generate data using LHS
Sw = 150 + (200-150) * lhsdesign(n, 1);
Wfw = 220 + (300-220) * lhsdesign(n, 1);
A = 6 + (10-6) * lhsdesign(n, 1);
delta = deg2rad(-10 + (20) * lhsdesign(n, 1));  % Convert degrees to radians
q = 16 + (45-16) * lhsdesign(n, 1);
lambda = 0.5 + (1-0.5) * lhsdesign(n, 1);
tc = 0.08 + (0.18-0.08) * lhsdesign(n, 1);
Nz = 2.5 + (6-2.5) * lhsdesign(n, 1);
Wdg = 1700 + (2500-1700) * lhsdesign(n, 1);
Wp = 0.025 + (0.08-0.025) * lhsdesign(n, 1);

% Calculate the wing weight output for each data point
y = zeros(n, 1);
for i = 1:n
    y(i) = wing_weight(Sw(i), Wfw(i), A(i), delta(i), q(i), lambda(i), tc(i), Nz(i), Wdg(i), Wp(i));
end

% Concatenate into a data matrix
data = [Sw Wfw A delta q lambda tc Nz Wdg Wp y];



% Display first few rows for verification
% disp(data(1:5, :));

% Assuming your data is in a matrix 'data'

% Perform PCA on your data
[coeff, score, latent, tsquared, explained] = pca(data(:,1:10));

% Display explained variance
disp('Explained Variance Ratios:');
disp(explained(1:2));

% Display factor loadings for the first two principal components
disp('Factor Loadings (First two components):');
disp(coeff(:,1:2));

%% Dataset

% Define component names
comp_names = {'Sw', 'Wfw', 'A', 'delta', 'q', 'lambda', 'tc', 'Nz', 'Wdg', 'Wp', 'y'};

% Create SOM dataset and normalize
sData = som_data_struct(data,'comp_names',comp_names); 
sData = som_normalize(sData,'range');

%% Initializing SOM Map Codebook Vectors (Linear Initialization)
[sMap]= modifiedsom_lininit1(sData,'lattice','hexa','msize',[20,20]);

%% Training SOM
[sMap,sTrain] = modifiedsom_batchtrain(sMap,sData,'sample_order','ordered','trainlen',500,...
     'radius_ini', 1.0, 'radius_fin',0.9);

%% Denormalizing the data
sMap=som_denormalize(sMap,sData);   
sData=som_denormalize(sData,'remove');

%% Visualization of SOM results (U Matrix and Component Planes)
figure(3) 
som_show(sMap, 'umat','all','comp', 'all');

%% iSOM Grid in function space (using just the first three components for visualization)
figure(4)
som_grid(sMap,'coord',sMap.codebook(:,[1 2 11]),'label',sMap.labels,'labelcolor','c','labelsize',10, 'marker','o','MarkerColor','k',...
    'MarkerSize',7,'linecolor', 'k');
hold on, scatter3(data(:,1),data(:,2),data(:,11),20,'ro','filled');
xlabel('Sw')
ylabel('Wfw')
zlabel('y')

% Display explained variance
disp('Explained Variance Ratios:');
disp(explained(1:2));

% Display factor loadings for the first two principal components
disp('Factor Loadings (First two components):');
disp(coeff(:,1:2));

%%
norm_data = som_normalize(sMap,'range');


% Extracting the codebook vectors
v1 = norm_data.codebook(:,1); % Sw
v2 = norm_data.codebook(:,2); % Wfw
v3 = norm_data.codebook(:,3); % A
v4 = norm_data.codebook(:,4); % delta
v5 = norm_data.codebook(:,5); % q
v6 = norm_data.codebook(:,6); % lambda
v7 = norm_data.codebook(:,7); % tc
v8 = norm_data.codebook(:,8); % Nz
v9 = norm_data.codebook(:,9); % Wdg
v10 = norm_data.codebook(:,10); % Wp
v11 = norm_data.codebook(:,11); % y (output)





% Compute slopes for each variable
Sw_slope = calculate_slope(v1);
Wfw_slope = calculate_slope(v2);
A_slope = calculate_slope(v3);
delta_slope = calculate_slope(v4);
q_slope = calculate_slope(v5);
lambda_slope = calculate_slope(v6);
tc_slope = calculate_slope(v7);
Nz_slope = calculate_slope(v8);
Wdg_slope = calculate_slope(v9);
Wp_slope = calculate_slope(v10);


% Project gradients of v11 onto the slopes of each feature
[Sw_pos, Sw_neg] = projected_slope(v11, Sw_slope);
[Wfw_pos, Wfw_neg] = projected_slope(v11, Wfw_slope);
[A_pos, A_neg] = projected_slope(v11, A_slope);
[delta_pos, delta_neg] = projected_slope(v11, delta_slope);
[q_pos, q_neg] = projected_slope(v11, q_slope);
[lambda_pos, lambda_neg] = projected_slope(v11, lambda_slope);
[tc_pos, tc_neg] = projected_slope(v11, tc_slope);
[Nz_pos, Nz_neg] = projected_slope(v11, Nz_slope);
[Wdg_pos, Wdg_neg] = projected_slope(v11, Wdg_slope);
[Wp_pos, Wp_neg] = projected_slope(v11, Wp_slope);

% Aggregate all the results into a new variable
results.positive_correls = [Sw_pos, Wfw_pos, A_pos, delta_pos, q_pos, lambda_pos, tc_pos, Nz_pos, Wdg_pos, Wp_pos];
results.negative_correls = [Sw_neg, Wfw_neg, A_neg, delta_neg, q_neg, lambda_neg, tc_neg, Nz_neg, Wdg_neg, Wp_neg];
disp(results.positive_correls)
disp(results.negative_correls)

%%
features = {'Sw', 'Wfw', 'A', 'delta', 'q', 'lambda', 'tc', 'Nz', 'Wdg', 'Wp'};
figure;

subplot(2, 1, 1);
hold on;
barh(1:10, -results.negative_correls, 'r');
barh(1:10, results.positive_correls, 'g');
yticks(1:10); yticklabels(features); xlabel('Correlation');
% legend('Negative', 'Positive');
grid on; hold off;

subplot(2, 1, 2);
barh(1:10, abs(results.positive_correls) + abs(results.negative_correls), 'b');
yticks(1:10); yticklabels(features); xlabel('Total Magnitude');
grid on;



%%




function slope = calculate_slope(codebook_vec)
    data_ = reshape(codebook_vec, [20, 20])
    data_ = flipud(data_);
    [dZdx, dZdy] = gradient(data_);
    avg_dZdx = mean(dZdx(2:end-1, 2:end-1), 'all');
    avg_dZdy = mean(dZdy(2:end-1, 2:end-1), 'all');
    slope = [avg_dZdx, avg_dZdy];
end



function [positive_correl, negative_correl] = projected_slope(codebook_vec, input_slope)

    % Reshape and flip the codebook vector
    data_ = reshape(codebook_vec, [20, 20]);
    data_ = flipud(data_);
    
    % Calculate gradient
    [dZdx, dZdy] = gradient(data_);

    % Normalize the gradients
    grad_magnitudes = sqrt(dZdx.^2 + dZdy.^2);
    dZdx = dZdx ./ grad_magnitudes;
    dZdy = dZdy ./ grad_magnitudes;

    disp([dZdx, dZdy])

    % Normalize the input slope to get a unit vector
    magnitude = norm(input_slope); 
    unit_slope = input_slope / magnitude;
    
    % Initialize variables to hold the sum of vectors
    pos_sum = [0, 0];
    neg_sum = [0, 0];
    
    % Project dZdx and dZdy onto the input slope
    for i = 2:19
        for j = 2:19
            gradient_vec = [dZdx(i, j), dZdy(i, j)]; % make it a row vector
            dot_product = dot(gradient_vec, unit_slope);
            
            % Project gradient vector onto the unit slope
            projected_vec = (dot_product) * unit_slope;
            
            if dot_product > 0
                pos_sum = pos_sum + projected_vec;
            else
                neg_sum = neg_sum + projected_vec;
            end
        end
        
    end
    
    % Compute the magnitudes for vectors in each direction
    positive_correl = norm(pos_sum)/324;
    negative_correl = norm(neg_sum)/324;
    
end




function [positive_correl, negative_correl] = correl(codebook_vec1, codebook_vec2)

    % Reshape and flip the first codebook vector
    data1 = reshape(codebook_vec1, [20, 20]);
    data1 = flipud(data1);

    % Calculate gradient for the first codebook vector
    [dZdx1, dZdy1] = gradient(data1);

    % Normalize the gradients for the first codebook vector
    grad_magnitudes1 = sqrt(dZdx1.^2 + dZdy1.^2);
    dZdx1 = dZdx1 ./ grad_magnitudes1;
    dZdy1 = dZdy1 ./ grad_magnitudes1;

    % Reshape and flip the second codebook vector
    data2 = reshape(codebook_vec2, [20, 20]);
    data2 = flipud(data2);

    % Calculate gradient for the second codebook vector
    [dZdx2, dZdy2] = gradient(data2);

    % Normalize the gradients for the second codebook vector
    grad_magnitudes2 = sqrt(dZdx2.^2 + dZdy2.^2);
    dZdx2 = dZdx2 ./ grad_magnitudes2;
    dZdy2 = dZdy2 ./ grad_magnitudes2;

    % Initialize variables to hold the sum of vectors
    pos_sum = [0, 0];
    neg_sum = [0, 0];

    % Project dZdx1 and dZdy1 onto dZdx2 and dZdy2
    for i = 2:19
        for j = 2:19
            gradient_vec1 = [dZdx1(i, j), dZdy1(i, j)]; % make it a row vector
            gradient_vec2 = [dZdx2(i, j), dZdy2(i, j)]; % make it a row vector

            dot_product = dot(gradient_vec1, gradient_vec2);

            % Project gradient vector onto the other gradient vector
            projected_vec = (dot_product) * gradient_vec2;

            if dot_product > 0
                pos_sum = pos_sum + projected_vec;
            else
                neg_sum = neg_sum + projected_vec;
            end
        end

    end

    % Compute the magnitudes for vectors in each direction
    positive_correl = norm(pos_sum) / 324;
    negative_correl = norm(neg_sum) / 324;

end






function fx = wing_weight(Sw, Wfw, A, delta, q, lambda, tc, Nz, Wdg, Wp)
    % Wing weight function based on the provided formula
    fx = 0.0368 * Sw^0.758 * Wfw^0.0035 ...
        * (A/cos(delta)^2)^0.6 * q^0.006 * lambda^0.04 ...
        * (100 * tc / cos(delta))^-0.3 ...
        * (Nz * Wdg)^0.49 ...
        + Sw * Wp;
end