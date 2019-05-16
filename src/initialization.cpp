#include "svo/initialization.hpp"

#include <vikit/homography.h>

namespace dr3 {

namespace init {

InitHelper::InitHelper(const FramePtr &_frame_ref, float sigma, int iterations) {
    auto cam = _frame_ref->get_cam();
    mK = dynamic_cast<Pinhole*>(cam)->K();

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}

bool InitHelper::Initialize(const FramePtr &frame_cur, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated) {
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());
    mvbMatched1.resize(mvKeys1.size());
    for(size_t i=0, iend=vMatches12.size();i<iend; i++) {
        if(vMatches12[i]>=0) {
            mvMatches12.emplace_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        } else {
            mvbMatched1[i]=false;
        }
    }

    DLOG(INFO) << "Ref frame key points (mvKeys1) count: " << mvKeys1.size();
    DLOG(INFO) << "Cur frame key points (mvKeys2) count: " << mvKeys2.size();
    DLOG(INFO) << "Initial number of matches : " << mvMatches12.size();

    const int N = mvMatches12.size();
    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
        vAllIndices.push_back(i);

    // Generate sets of 8 points for each RANSAC iteration
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    // TODO: Change to C++11 random library
    struct timeval _time;
    gettimeofday(&_time, nullptr);
    srand(time(nullptr));
    for(int it=0; it<mMaxIterations; it++) {
        vAvailableIndices = vAllIndices;
        // Select a minimum set
        for(size_t j=0; j<8; j++) {
            int randi = rand() % (vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];
            mvSets[it][j] = idx;
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    vector<bool> vbMatchesInliersF;
    float SF;
    cv::Mat F;

    DLOG(INFO) << "Estimating fundamental matrix";
    FindFundamental(ref(vbMatchesInliersF), ref(SF), ref(F));

    if (!F.data) {
        LOG(FATAL) << "Fundamental matrix not estimated!";
    }

    DLOG(INFO) << "Generating 3D points for the inliers";
    return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
}

void InitHelper::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21) {
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1, vPn1, T1);
    Normalize(mvKeys2, vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    DLOG(INFO) << "Performing " << mMaxIterations << " RANSAC iters to estimate F";
    for(int it=0; it<mMaxIterations; it++) {
        // Select a minimum set
        for(int j=0; j<8; j++) {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        // Estimate F
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        if (!Fn.data) {
            DLOG(WARNING) << "Fundamental matrix estimation failed";
        }

        // Count the number of inliers
        F21i = T2t*Fn*T1;
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        // Select the best Fundamental matrix
        if(currentScore>score) {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
    DLOG(INFO) << "Estimated Fundamental matrix with max. chisquare score: " << score;
}

cv::Mat InitHelper::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2) {
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

float InitHelper::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma) {

    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

bool InitHelper::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                               cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for (size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    DecomposeE(E21,R1,R2,t);

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    DLOG(INFO) << "Generating 4 hypothesis for initial camera estimation";
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood) {
        LOG(WARNING) << "Not enough triangulated points [min: " << nMinGood << "] obtained: [" << maxGood << "]";
        return false;
    }
    if(nsimilar>1) {
        LOG(WARNING) << "Not enough parallax between the two views";
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1) {
        if(parallax1>minParallax) {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    } else if(maxGood==nGood2) {
        if(parallax2>minParallax) {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    } else if(maxGood==nGood3) {
        if(parallax3>minParallax) {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    } else if(maxGood==nGood4) {
        if(parallax4>minParallax) {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    LOG(WARNING) << "No solution could be recovered";
    return false;
}

void InitHelper::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D) {
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

void InitHelper::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T) {
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

int InitHelper::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                         const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                         const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax) {
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

void InitHelper::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t) {
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

Init::Init() : initializer(nullptr) {
}

Result Init::process(FramePtr &frame) {
    Features new_features;
    feature_detection::FastDetector detector(frame->_img_pyr[0].cols,
                                             frame->_img_pyr[0].rows,
                                             Config::cell_size(),
                                             Config::n_pyr_levels());
    detector.detect(frame, frame->_img_pyr,
                    Config::min_harris_corner_score(),
                    new_features);

    if (new_features.size() < 100) {
        if (!initializer) {
            LOG(WARNING) << "NOT ENOUGH FEATURES ARE DETECTED IN REFERENCE FRAME";
        } else {
            LOG(WARNING) << "NOT ENOUGH FEATURES ARE DETECTED IN CURRENT FRAME";
        }

        initializer = static_cast<InitHelper*>(nullptr);
        fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
        return Result::FAILED;
    }

    if (!initializer) {
        // Add the reference frame
        frame_ref = frame;

        _kps_ref.clear(); _kps_ref.reserve(new_features.size());
        _pts_ref.clear(); _pts_ref.reserve(new_features.size());
        mvbPrevMatched.clear(); mvbPrevMatched.reserve(new_features.size());
        std::for_each(new_features.begin(), new_features.end(), [&](Feature *ftr) {
            cv::Point2f kpt(ftr->px[0], ftr->px[1]);
            _kps_ref.emplace_back(kpt);
            _pts_ref.push_back(ftr->f);
            mvbPrevMatched.emplace_back(kpt);
            delete ftr;
        });

        _kps_cur.insert(_kps_cur.begin(), _kps_ref.begin(), _kps_ref.end());
        initializer = new InitHelper(frame, 1.0f, 200);
        mvIniMatches.reserve(_kps_ref.size());
        fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

        return Result::SUCCESS;
    } else {
        // Add the current frame and generate the initial map
        frame_cur = frame;
        DLOG(INFO) << "Features detected in reference frame: " << _kps_ref.size();
        DLOG(INFO) << "Features detected in current frame: " << _kps_cur.size();
        std::for_each(new_features.begin(), new_features.end(), [&](Feature *ftr) {
            delete ftr;
        });

        const int klt_win_size = 30;
        const int klt_max_iter = 1000;
        const double klt_eps = 1e-3;
        vector<uchar> status;
        vector<float> error;
        cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                  klt_max_iter, klt_eps);

        if (!frame_ref->_img_pyr[0].data) {
            LOG(WARNING) << "Invalid reference image";
        }
        if (!frame_cur->_img_pyr[0].data) {
            LOG(WARNING) << "Invalid current image";
        }

        cv::calcOpticalFlowPyrLK(frame_ref->_img_pyr[0],
                                 frame_cur->_img_pyr[0],
                                 _kps_ref, _kps_cur,
                                 status, error,
                                 cv::Size2i(klt_win_size, klt_win_size),
                                 4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

        auto kps_ref_itr = _kps_ref.begin();
        auto kps_cur_itr = _kps_cur.begin();
        auto pts_ref_itr = _pts_ref.begin();
        _disparities.clear(); _disparities.reserve(_kps_cur.size());
        _pts_cur.clear(); _pts_cur.reserve(_kps_cur.size());
        size_t outlier_count = 0;
        for (size_t i = 0; kps_ref_itr != _kps_ref.end(); ++i) {
            if (!status[i]) {
                kps_ref_itr = _kps_ref.erase(kps_ref_itr);
                kps_cur_itr = _kps_cur.erase(kps_cur_itr);
                pts_ref_itr = _pts_ref.erase(pts_ref_itr);
                ++outlier_count;
                continue;
            }
            _disparities.push_back(Vector2d(kps_ref_itr->x - kps_cur_itr->x,
                                            kps_ref_itr->y - kps_cur_itr->y).norm());
            _pts_cur.push_back(frame_cur->_cam->cam2world(kps_cur_itr->x, kps_cur_itr->y));
            ++kps_ref_itr;
            ++kps_cur_itr;
            ++pts_ref_itr;
        }

        // Update the frame's keypoints
        vector<cv::KeyPoint> &mvKeys1 = initializer->mutable_keys_ref();
        mvKeys1.clear(); mvKeys1.reserve(_kps_ref.size());
        std::for_each(_kps_ref.begin(), _kps_ref.end(), [&](cv::Point2f pt) {
            mvKeys1.emplace_back(cv::KeyPoint(pt, 2.0));
        });
        vector<cv::KeyPoint> &mvKeys2 = initializer->mutable_keys_cur();
        mvKeys2.clear(); mvKeys2.reserve(_kps_cur.size());
        std::for_each(_kps_cur.begin(), _kps_cur.end(), [&](cv::Point2f pt) {
            mvKeys2.emplace_back(cv::KeyPoint(pt, 2.0));
        });
        mvIniMatches.reserve(mvKeys1.size());
        for (size_t i = 0; i < mvKeys1.size(); ++i)
            mvIniMatches.emplace_back(i);

        DLOG(INFO) << "Outlier count from optical flow: " << outlier_count;
        DLOG(INFO) << "Average disparity: " << accumulate(_disparities.begin(), _disparities.end(), 0.0) / _disparities.size() << "px";
        DLOG(INFO) << "Total matches between ref and cur frames: " << mvIniMatches.size();
        if (mvIniMatches.size() < 100) {
            LOG(WARNING) << "Very few matches (<100) detected";
        }

        cv::Mat R, t;
        vector<bool> triangulated;
        if (!initializer->Initialize(frame_cur, mvIniMatches, R, t, mvIniP3D, triangulated)) {
            return Result::FAILED;
        }

        int triangulated_count = accumulate(triangulated.begin(), triangulated.end(), 0);
        if (triangulated_count < 100) {
            LOG(WARNING) << "< 100 inliers are triangulated";
            return Result::FAILED;
        }
        _xyz_in_world.reserve(triangulated.size());
        for (int i = 0; i < triangulated.size(); ++i) {
            if (!triangulated[i]) {
                continue;
            } else {
                Vector3d pt;
                pt(0) = mvIniP3D[i].x;
                pt(1) = mvIniP3D[i].y;
                pt(2) = mvIniP3D[i].z;
                _xyz_in_world.emplace_back(pt);
            }
        }

        Matrix3d _R; Vector3d _t;
        _R(0, 0) = R.at<float>(0, 0);
        _R(0, 1) = R.at<float>(0, 1);
        _R(0, 2) = R.at<float>(0, 2);
        _R(1, 0) = R.at<float>(1, 0);
        _R(1, 1) = R.at<float>(1, 1);
        _R(1, 2) = R.at<float>(1, 2);
        _R(2, 0) = R.at<float>(2, 0);
        _R(2, 1) = R.at<float>(2, 1);
        _R(2, 2) = R.at<float>(2, 2);
        _t(0) = t.at<float>(0);
        _t(1) = t.at<float>(1);
        _t(2) = t.at<float>(2);
        _T_cur_from_ref = Sophus::SE3(_R, _t); // TODO: T or T.inv()?

        _inliers.reserve(triangulated.size());
        for (size_t i = 0; i < triangulated.size(); ++i)
            if (triangulated[i])
                _inliers.emplace_back(i);

        // Rescale the map such that the mean scene depth is equal to the specified scale
        vector<double> depth_vec;
        for (auto itr: mvIniP3D) {
            depth_vec.push_back(itr.z);
        }

        double scene_depth_median = vk::getMedian(depth_vec);
        double scale = Config::map_scale() / scene_depth_median;

        frame_cur->_T_f_w = _T_cur_from_ref * frame_ref->_T_f_w;
        frame_cur->_T_f_w.translation() =
                -frame_cur->_T_f_w.rotation_matrix()*(frame_ref->pos() + scale*(frame_cur->pos() - frame_ref->pos()));

        // Create a 3D point and add the observations
        SE3 T_world_cur = frame_cur->_T_f_w.inverse();
        size_t count = 0;
        for (auto itr: _inliers) {
            Vector2d px_cur(_kps_cur[itr].x, _kps_cur[itr].y);
            Vector2d px_ref(_kps_ref[itr].x, _kps_ref[itr].y);
            if (frame_ref->_cam->is_in_frame(px_cur.cast<int>(), 10) &&
                frame_ref->_cam->is_in_frame(px_ref.cast<int>(), 10) &&
                _xyz_in_world[itr].z() > 0) {
                // not necessary? (_xyz_in_cur in world?)
                Vector3d pos = _xyz_in_world[itr]*scale; // TODO: correct? no transformation?
                auto new_point = new Point(pos);

                auto ftr_cur(new Feature(frame_cur, new_point, px_cur, _pts_cur[itr], 0));
                frame_cur->add_observation(ftr_cur);
                new_point->add_observation(frame_cur);

                auto ftr_ref(new Feature(frame_ref, new_point, px_ref, _pts_ref[itr], 0));
                frame_ref->add_observation(ftr_ref);
                new_point->add_observation(frame_ref);
                ++count;
            }
        }
        LOG(INFO) << "Generated initial map with " << count << " points";
        return Result::SUCCESS;
    }
}

} // namespace init

} // namespace dr3
