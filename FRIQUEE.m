%% An example script to demonstrate FRIQUEE feature extraction and prediction of image quality.
% This script performs two tasks:
% 1 : Extracts FRIQUEE features
% 2 : Loads a learned model trained on all the images of LIVE Challenge Database and predicts the quality of the given example image. The quality is predicted on a scale of 0-100, where 0 represents the worst score and 1 represents the best score.
%Dependencies
%  The assumption here is that you have libsvm installed and
% svmpredict binary built

% Modification made by:
% Jose Alejandro Ledesma and Stidl Alfonso torres
% Electronic Engineers 
% For graduate work "Evaluation of no-reference quality prediction metrics in videos impaired by authentic distortions"
% Pontificia Universidad Javeriana Cali, Santiago de Cali 2019-2020
% Supervised by:
% Hernán Darío Benítez Restrepo
% Roger Alfonso Gómez Nieto


clear;
addpath('include/');
addpath('include/C_DIIVINE');
addpath('D:\JAVERIANA\THESIS\ALGORITMO1_FRIQUEE\libsvm-3.24\matlab');
addpath('src/');
addpath('data'); 
load('data/friqueeLearnedModel.mat');


% Direccion de la carpeta de videos
videos_path = 'D:\JAVERIANA\THESIS\BASES DE DATOS\VIDEOS\KoNViD_1k_videos\';

% Direccion de la carpeta donde se guardaran las caracteristicas extraidas 
Feature_path= 'D:\JAVERIANA\THESIS\ALGORITMO1_FRIQUEE\FRIQUEE_Release\Resultados konvid\Features\';

% Direccion de la carpeta donde se guardaran los puntajes objetivos extraidos 
Scorepath= 'D:\JAVERIANA\THESIS\ALGORITMO1_FRIQUEE\FRIQUEE_Release\Resultados konvid\Scores\';

list=dir(videos_path)
for i=1172:length(list)    
    list(i).name

    % Se le pasa la direccion del video a evaluar 
    video = strcat(videos_path,list(i).name);    
    v = VideoReader(video);
    vid_frames = read(v);  %read all frames
    NumberofFrame=size(vid_frames,4); %Numero de frames del video
    Matriz_de_caracteristicas=zeros( NumberofFrame,560); % Se crea una matriz del tamaño indicado para cada video equivalente a numero_de_framesX560 (numero de columnas de caracteristicas de FRIQUEE)
    parfor j=1:NumberofFrame
        i
        j
        frame_vid=vid_frames(:,:,:,j);

        % Extract FRIQUEE-ALL features of this image
        testFriqueeFeats = extractFRIQUEEFeatures(frame_vid);
        Matriz_de_caracteristicas(j,:)=testFriqueeFeats.friqueeALL;
        
     
        % Scale the features of the test image accordingly.
        % The minimum and the range are computed on features of all the images of
        % LIVE Challenge Database  
        testFriqueeALL = testFeatNormalize(testFriqueeFeats.friqueeALL, friqueeLearnedModel.trainDataMinVals, friqueeLearnedModel.trainDataRange);

        qualityScore(i,j)= svmpredict (0, double(testFriqueeALL), friqueeLearnedModel.trainModel, '-b 1 -q');
    end 

    % Se almacena los resultados tanto los puntajes como caracteristicas en la archivos Mat en las rutas indicadas
    Fvideo=strcat(Feature_path,list(i).name);
    Fvideo2=strcat(Fvideo,'.mat');
    Svideo=strcat(Scorepath,list(i).name);
    Svideo2=strcat(Svideo,'.mat');
    save(Fvideo2,'Matriz_de_caracteristicas','-v7.3');  
    save (Svideo2,'qualityScore','-v7.3');    
end
