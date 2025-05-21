#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

// Physical parameters
const double Ra = 2e2;
const double Pr = 50;
const double nu = std::sqrt(Pr / Ra);
const double kappa = nu / Pr;

// Grid parameters
const int nx = 320, ny = 200;
const double Lx = 4.0, Ly = 0.4;
const double dx = Lx / nx, dy = Ly / ny;

// Time stepping
const double t_max = 0.1;
const int nt = 30000;
const double dt = t_max / nt;

// Video parameters
const int fps = 25;             // frames per second for MP4
const int frame_interval = 25;  // capture a frame every 25 steps

// SOR solver parameters
const int max_iter = 20000;
const double tol = 1e-6;
const double omega = 1.7;

// Scaling factor for higher resolution
const int scale = 5;  // scale factor for width and height

// Print a console progress bar
void printProgress(int step, int total) {
    const int barWidth = 50;
    float progress = (step + 1) / static_cast<float>(total);
    int pos = static_cast<int>(barWidth * progress);

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(progress * 100) << "\n";
    std::cout.flush();
}

// Apply boundary conditions for psi, xi, Theta
void applyBC(cv::Mat& psi, cv::Mat& xi, cv::Mat& Theta) {
    psi.row(0).setTo(0);
    psi.row(ny).setTo(0);
    psi.col(0) = psi.col(nx - 1);
    psi.col(nx).copyTo(psi.col(1));

    for (int j = 0; j <= nx; ++j) {
        xi.at<double>(0, j)   = -2.0 * psi.at<double>(1, j)    / (dy * dy);
        xi.at<double>(ny, j)  = -2.0 * psi.at<double>(ny-1, j) / (dy * dy);
    }
    xi.col(0) = xi.col(nx - 1);
    xi.col(nx).copyTo(xi.col(1));

    for (int j = 0; j <= nx; ++j) {
        Theta.at<double>(0, j)  = 1.0;
        Theta.at<double>(ny, j) = 0.0;
    }
    Theta.col(0) = Theta.col(nx - 1);
    Theta.col(nx).copyTo(Theta.col(1));
}

// SOR solver for stream function psi
void solveStream(cv::Mat& psi, const cv::Mat& xi) {
    double ax = dx * dx;
    double ay = dy * dy;
    double den = 2.0 * (ax + ay);

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_err = 0.0;
        for (int i = 1; i < ny; ++i) {
            for (int j = 1; j < nx; ++j) {
                double val = (psi.at<double>(i+1,j) + psi.at<double>(i-1,j)) * ax / den
                           + (psi.at<double>(i,j+1) + psi.at<double>(i,j-1)) * ay / den
                           + xi.at<double>(i,j) * ax * ay / den;
                double psi_new = omega * val + (1.0 - omega) * psi.at<double>(i,j);
                max_err = std::max(max_err, std::abs(psi_new - psi.at<double>(i,j)));
                psi.at<double>(i,j) = psi_new;
            }
        }
        if (max_err < tol) break;
    }
}

int main() {
    std::cout << "Starting simulation..." << std::endl;

    // Initialize fields
    cv::Mat psi   = cv::Mat::zeros(ny+1, nx+1, CV_64F);
    cv::Mat xi    = cv::Mat::zeros(ny+1, nx+1, CV_64F);
    cv::Mat Theta = cv::Mat::zeros(ny+1, nx+1, CV_64F);
    cv::Mat u     = cv::Mat::zeros(ny+1, nx+1, CV_64F);
    cv::Mat v     = cv::Mat::zeros(ny+1, nx+1, CV_64F);
    cv::Mat phi   = cv::Mat::zeros(ny+1, nx+1, CV_64F);

    // Apply initial boundary conditions and solve initial stream function
    applyBC(psi, xi, Theta);
    solveStream(psi, xi);

    // RNG for temperature perturbations
    std::mt19937 rng(0);
    std::normal_distribution<double> dist(0.0, 1e-6);

    // Prepare storage for video frames (unchanged) and snapshots
    std::vector<cv::Mat> frames;
    frames.reserve(nt / frame_interval + 1);

    const int n_snapshots = 10;
    const int save_interval = nt / n_snapshots;
    std::vector<cv::Mat> snapshots;
    snapshots.reserve(n_snapshots);

    // Main time-stepping loop
    for (int step = 0; step < nt; ++step) {
        // 1) Compute velocities from psi
        for (int i = 1; i < ny; ++i)
            for (int j = 1; j < nx; ++j) {
                u.at<double>(i,j) = (psi.at<double>(i+1,j) - psi.at<double>(i-1,j)) / (2.0 * dy);
                v.at<double>(i,j) = -(psi.at<double>(i,j+1) - psi.at<double>(i,j-1)) / (2.0 * dx);
            }
        // Velocity BCs
        u.row(0).setTo(0);      u.row(ny).setTo(0);
        v.row(0).setTo(0);      v.row(ny).setTo(0);
        u.col(0) = u.col(nx-1); u.col(nx).copyTo(u.col(1));
        v.col(0) = v.col(nx-1); v.col(nx).copyTo(v.col(1));

        // 2) Vorticity (xi) update
        cv::Mat xi_new = xi.clone();
        for (int i = 1; i < ny; ++i) for (int j = 1; j < nx; ++j) {
            int alpha = u.at<double>(i,j) > 0;
            int beta  = v.at<double>(i,j) > 0;
            double xi_x = u.at<double>(i,j) * (
                alpha * (xi.at<double>(i,j) - xi.at<double>(i,j-1)) +
                (1-alpha) * (xi.at<double>(i,j+1) - xi.at<double>(i,j))
            ) / dx;
            double xi_y = v.at<double>(i,j) * (
                beta * (xi.at<double>(i,j) - xi.at<double>(i-1,j)) +
                (1-beta) * (xi.at<double>(i+1,j) - xi.at<double>(i,j))
            ) / dy;
            double xi_xx = (xi.at<double>(i,j+1) - 2*xi.at<double>(i,j) + xi.at<double>(i,j-1)) / (dx*dx);
            double xi_yy = (xi.at<double>(i+1,j) - 2*xi.at<double>(i,j) + xi.at<double>(i-1,j)) / (dy*dy);
            xi_new.at<double>(i,j) = xi.at<double>(i,j)
                + dt * ( - (xi_x + xi_y)
                         + nu * (xi_xx + xi_yy)
                         + Ra * (Theta.at<double>(i,j+1) - Theta.at<double>(i,j-1)) / (2.0*dx)
                       );
        }
        xi = xi_new;

        // 3) Temperature (Theta) update
        cv::Mat T_new = Theta.clone();
        for (int i = 1; i < ny; ++i) for (int j = 1; j < nx; ++j) {
            double Tx  = (Theta.at<double>(i,j+1) - Theta.at<double>(i,j-1)) / (2.0*dx);
            double Ty  = (Theta.at<double>(i+1,j) - Theta.at<double>(i-1,j)) / (2.0*dy);
            double Txx = (Theta.at<double>(i,j+1) - 2*Theta.at<double>(i,j) + Theta.at<double>(i,j-1)) / (dx*dx);
            double Tyy = (Theta.at<double>(i+1,j) - 2*Theta.at<double>(i,j) + Theta.at<double>(i-1,j)) / (dy*dy);
            T_new.at<double>(i,j) = Theta.at<double>(i,j)
                + dt * ( - u.at<double>(i,j)*Tx
                         - v.at<double>(i,j)*Ty
                         + kappa * (Txx + Tyy)
                       );
        }
        Theta = T_new;

        // Add small noise every 50 steps
        // if (step % 50 == 0) {
        //     for (int i = 1; i < ny; ++i)
        //         for (int j = 1; j < nx; ++j)
        //             Theta.at<double>(i,j) += dist(rng);
        //
        // }

        // Re-apply BCs and re-solve stream function
        applyBC(psi, xi, Theta);
        solveStream(psi, xi);

        // Compute flow angle phi
        for (int i = 0; i <= ny; ++i)
            for (int j = 0; j <= nx; ++j)
                phi.at<double>(i,j) = std::atan2(v.at<double>(i,j), u.at<double>(i,j));

        // --- Capture frame for video (unchanged) ---
        if (step % frame_interval == 0) {
            cv::Mat img;
            Theta.convertTo(img, CV_8U, 255.0);
            cv::applyColorMap(img, img, cv::COLORMAP_INFERNO);
            cv::Mat img_big;
            cv::resize(img, img_big, cv::Size((nx+1)*scale, (ny+1)*scale), 0, 0, cv::INTER_CUBIC);
            cv::rotate(img_big, img_big, cv::ROTATE_180);
            frames.push_back(img_big);
            printProgress(step, nt);
        }

        cv::Size finalSz((nx+1)*scale, (ny+1)*scale);

        // --- Capture snapshot with 3 enhanced plots ---
        if (step % save_interval == 0) {
            // 1) Temperature field (как раньше)
            cv::Mat T8, Tcolor;
            Theta.convertTo(T8, CV_8U, 255.0);
            cv::applyColorMap(T8, Tcolor, cv::COLORMAP_INFERNO);

            // 2) Vorticity ξ: heatmap + сглаженные контуры + стрелки
            // -------------------------
            // a) heatmap
            cv::Mat xi_smooth, xi8, xi_heat;
            cv::GaussianBlur(xi, xi_smooth, cv::Size(7,7), 0);
            cv::normalize(xi_smooth, xi8, 0, 255, cv::NORM_MINMAX);
            xi8.convertTo(xi8, CV_8U);
            cv::applyColorMap(xi8, xi_heat, cv::COLORMAP_BONE);  // CV_8UC3

            // b) контуры (на том же разрешении)
            cv::Mat xi_mask, xi_contours_mat = cv::Mat::zeros(xi8.size(), CV_8UC3);
            double minv, maxv;
            cv::minMaxLoc(xi_smooth, &minv, &maxv);
            int levels = 12;
            for (int k = 1; k < levels; ++k) {
                double lvl = minv + k*(maxv-minv)/levels;
                cv::threshold(xi_smooth, xi_mask, lvl, 255, cv::THRESH_BINARY);
                xi_mask.convertTo(xi_mask, CV_8U);
                std::vector<std::vector<cv::Point>> ctr;
                cv::findContours(xi_mask, ctr, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
                for (auto &c : ctr)
                    cv::polylines(xi_contours_mat, c, true, cv::Scalar(255,255,255), 1, cv::LINE_AA);
            }

            // c) overlay heatmap + контуры
            cv::Mat xi_overlay;
            cv::addWeighted(xi_heat,    0.7,
                            xi_contours_mat, 0.3,
                            0, xi_overlay);   // CV_8UC3

            // d) стрелки скорости
            int stride = 15;
            for (int i = 0; i <= ny; i += stride) {
                for (int j = 0; j <= nx; j += stride) {
                    double uu = u.at<double>(i,j), vv = v.at<double>(i,j);
                    double mag = std::hypot(uu,vv);
                    if (mag < 1e-6) continue;
                    cv::Point2f p1(j, i),
                                p2(j + uu/mag*stride*0.5f,
                                   i - vv/mag*stride*0.5f);
                    cv::arrowedLine(xi_overlay, p1, p2, cv::Scalar(255,255,255), 1, cv::LINE_AA, 0, 0.3);
                }
            }

            // 3) Stream-function ψ: сглаживаем + контуры + стрелки
            // ----------------------------------------------------
            cv::Mat psi_smooth, psi_up, psi_contours_mat = cv::Mat::zeros(psi.size(), CV_8UC3);
            cv::GaussianBlur(psi, psi_smooth, cv::Size(7,7), 0);
            cv::normalize(psi_smooth, psi_up, 0, 255, cv::NORM_MINMAX);
            psi_up.convertTo(psi_up, CV_8U);

            double pmin, pmax;
            cv::minMaxLoc(psi_smooth, &pmin, &pmax);
            for (int k = 1; k < levels; ++k) {
                double lvl = pmin + k*(pmax-pmin)/levels;
                cv::Mat mask;
                cv::threshold(psi_smooth, mask, lvl, 255, cv::THRESH_BINARY);
                mask.convertTo(mask, CV_8U);
                std::vector<std::vector<cv::Point>> ctr;
                cv::findContours(mask, ctr, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
                for (auto &c : ctr)
                    cv::polylines(psi_contours_mat, c, true, cv::Scalar(200,200,255), 1, cv::LINE_AA);
            }
            // стрелки
            for (int i = 0; i <= ny; i += stride) {
                for (int j = 0; j <= nx; j += stride) {
                    double uu = u.at<double>(i,j), vv = v.at<double>(i,j);
                    double mag = std::hypot(uu,vv);
                    if (mag < 1e-6) continue;
                    cv::Point2f p1(j,i),
                                p2(j + uu/mag*stride*0.5f,
                                   i - vv/mag*stride*0.5f);
                    cv::arrowedLine(psi_contours_mat, p1, p2, cv::Scalar(200,200,255), 1, cv::LINE_AA, 0, 0.3);
                }
            }

            // 4) Приводим всё к finalSz и CV_8UC3, поворачиваем, подписываем
            // -------------------------------------------------------------
            cv::Mat Tf, xiF, psiF;
            cv::resize(Tcolor,      Tf,  finalSz, 0,0, cv::INTER_LANCZOS4);
            cv::resize(xi_overlay,  xiF, finalSz, 0,0, cv::INTER_LANCZOS4);
            cv::resize(psi_contours_mat, psiF, finalSz, 0,0, cv::INTER_LANCZOS4);

            cv::rotate(Tf,  Tf,  cv::ROTATE_180);
            cv::rotate(xiF, xiF, cv::ROTATE_180);
            cv::rotate(psiF,psiF,cv::ROTATE_180);

            int font       = cv::FONT_HERSHEY_SIMPLEX;
            double s       = 0.6;
            int th         = 1;
            cv::Scalar w(255,255,255);

            cv::putText(Tf,  "Temperature (Theta)",      cv::Point(10,20), font, s, w, th);
            cv::putText(xiF, "Vorticity (xi)",           cv::Point(10,20), font, s, w, th);
            cv::putText(psiF,"Stream function (psi)",    cv::Point(10,20), font, s, w, th);

            // 5) Склеиваем по горизонтали – теперь все три одной размерности и типа!
            std::vector<cv::Mat> parts = { Tf, xiF, psiF };
            cv::Mat snapshot_row;
            cv::hconcat(parts, snapshot_row);

            snapshots.push_back(snapshot_row);
            printProgress(step, nt);
        }
    }

    std::cout << std::endl;

    // Write MP4 video at high resolution (unchanged)
    cv::Size frameSize((nx+1)*scale, (ny+1)*scale);
    cv::VideoWriter writer("simulation.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps, frameSize);
    if (!writer.isOpened()) {
        std::cerr << "Error: cannot open MP4 writer\n";
        return -1;
    }
    for (auto& f : frames) writer << f;
    writer.release();
    std::cout << "Video saved as simulation.mp4" << std::endl;

    // Save the 10 snapshots to disk
    for (int i = 0; i < snapshots.size(); ++i) {
        std::ostringstream name;
        name << "snapshot_" << std::setw(2) << std::setfill('0') << i << ".png";
        cv::imwrite(name.str(), snapshots[i]);
    }
    std::cout << "Snapshots saved (10 images)" << std::endl;

    // Final composite image (unchanged)
    auto toColor = [&](const cv::Mat& M, int cmap){
        cv::Mat tmp, out;
        double mn, mx;
        cv::minMaxLoc(M, &mn, &mx);
        M.convertTo(tmp, CV_8U, 255.0/(mx-mn), -255.0*mn/(mx-mn));
        cv::applyColorMap(tmp, out, cmap);
        return out;
    };
    cv::Mat imgPsi   = toColor(psi,   cv::COLORMAP_OCEAN);
    cv::Mat imgXi    = toColor(xi,    cv::COLORMAP_BONE);
    cv::Mat imgTheta = toColor(Theta, cv::COLORMAP_INFERNO);
    cv::Mat imgPhi   = toColor(phi,   cv::COLORMAP_HSV);
    cv::Mat top, bottom, finalImg, finalBig;
    cv::hconcat(imgPsi, imgXi,   top);
    cv::hconcat(imgTheta,imgPhi, bottom);
    cv::vconcat(top, bottom, finalImg);
    cv::resize(finalImg, finalBig,
               cv::Size(finalImg.cols*scale, finalImg.rows*scale),
               0, 0, cv::INTER_LANCZOS4);
    cv::rotate(finalBig, finalBig, cv::ROTATE_180);
    cv::imwrite("final.png", finalBig);
    std::cout << "Final image saved as final.png" << std::endl;

    return 0;
}

// int main() {
//     std::cout << "Starting simulation..." << std::endl;
//
//     // Initialize fields
//     cv::Mat psi   = cv::Mat::zeros(ny+1, nx+1, CV_64F);
//     cv::Mat xi    = cv::Mat::zeros(ny+1, nx+1, CV_64F);
//     cv::Mat Theta = cv::Mat::zeros(ny+1, nx+1, CV_64F);
//     cv::Mat u     = cv::Mat::zeros(ny+1, nx+1, CV_64F);
//     cv::Mat v     = cv::Mat::zeros(ny+1, nx+1, CV_64F);
//     cv::Mat phi   = cv::Mat::zeros(ny+1, nx+1, CV_64F);
//
//     applyBC(psi, xi, Theta);
//     solveStream(psi, xi);
//
//     // RNG for Theta perturbations
//     std::mt19937 rng(0);
//     std::normal_distribution<double> dist(0.0, 1e-6);
//
//     std::vector<cv::Mat> frames;
//     frames.reserve(nt / frame_interval + 1);
//
//     for (int step = 0; step < nt; ++step) {
//         // Compute velocities
//         for (int i = 1; i < ny; ++i) for (int j = 1; j < nx; ++j) {
//             u.at<double>(i,j) = (psi.at<double>(i+1,j) - psi.at<double>(i-1,j)) / (2.0 * dy);
//             v.at<double>(i,j) =-(psi.at<double>(i,j+1) - psi.at<double>(i,j-1)) / (2.0 * dx);
//         }
//         // BC for u, v
//         u.row(0).setTo(0); u.row(ny).setTo(0);
//         v.row(0).setTo(0); v.row(ny).setTo(0);
//         u.col(0)=u.col(nx-1); u.col(nx).copyTo(u.col(1));
//         v.col(0)=v.col(nx-1); v.col(nx).copyTo(v.col(1));
//
//         // Vorticity update
//         cv::Mat xi_new = xi.clone();
//         for (int i = 1; i < ny; ++i) for (int j = 1; j < nx; ++j) {
//             int alpha = u.at<double>(i,j)>0;
//             int beta  = v.at<double>(i,j)>0;
//             double xi_x = u.at<double>(i,j)*(alpha*(xi.at<double>(i,j)-xi.at<double>(i,j-1))+(1-alpha)*(xi.at<double>(i,j+1)-xi.at<double>(i,j)))/dx;
//             double xi_y = v.at<double>(i,j)*(beta*(xi.at<double>(i,j)-xi.at<double>(i-1,j))+(1-beta)*(xi.at<double>(i+1,j)-xi.at<double>(i,j)))/dy;
//             double xi_xx= (xi.at<double>(i,j+1)-2*xi.at<double>(i,j)+xi.at<double>(i,j-1))/(dx*dx);
//             double xi_yy= (xi.at<double>(i+1,j)-2*xi.at<double>(i,j)+xi.at<double>(i-1,j))/(dy*dy);
//             xi_new.at<double>(i,j) = xi.at<double>(i,j) + dt*(-(xi_x+xi_y)+nu*(xi_xx+xi_yy)+Ra*(Theta.at<double>(i,j+1)-Theta.at<double>(i,j-1))/(2.0*dx));
//         }
//         xi = xi_new;
//
//         // Temperature update
//         cv::Mat T_new = Theta.clone();
//         for (int i = 1; i < ny; ++i) for (int j = 1; j < nx; ++j) {
//             double Tx = (Theta.at<double>(i,j+1)-Theta.at<double>(i,j-1))/(2.0*dx);
//             double Ty = (Theta.at<double>(i+1,j)-Theta.at<double>(i-1,j))/(2.0*dy);
//             double Txx= (Theta.at<double>(i,j+1)-2*Theta.at<double>(i,j)+Theta.at<double>(i,j-1))/(dx*dx);
//             double Tyy= (Theta.at<double>(i+1,j)-2*Theta.at<double>(i,j)+Theta.at<double>(i-1,j))/(dy*dy);
//             T_new.at<double>(i,j)=Theta.at<double>(i,j)+dt*(-u.at<double>(i,j)*Tx - v.at<double>(i,j)*Ty + kappa*(Txx+Tyy));
//         }
//         Theta = T_new;
//
//         if (step % 50 == 0) for (int i = 1; i < ny; ++i) for (int j = 1; j < nx; ++j) Theta.at<double>(i,j)+=dist(rng);
//
//         applyBC(psi, xi, Theta);
//         solveStream(psi, xi);
//
//         for (int i = 0; i <= ny; ++i) for (int j = 0; j <= nx; ++j)
//             phi.at<double>(i,j)=std::atan2(v.at<double>(i,j), u.at<double>(i,j));
//
//         if (step % frame_interval == 0) {
//             cv::Mat img;
//             Theta.convertTo(img, CV_8U, 255.0);
//             cv::applyColorMap(img, img, cv::COLORMAP_INFERNO);
//             cv::Mat img_big;
//             cv::resize(img, img_big, cv::Size((nx+1)*scale, (ny+1)*scale), 0,0, cv::INTER_LINEAR);
//             cv::rotate(img_big, img_big, cv::ROTATE_180);
//             frames.push_back(img_big);
//
//             printProgress(step, nt);
//
//         }
//     }
//     std::cout << std::endl;
//
//     // Write MP4 video at high resolution
//     cv::Size frameSize((nx+1)*scale, (ny+1)*scale);
//     cv::VideoWriter writer("simulation.mp4",
//         cv::VideoWriter::fourcc('m','p','4','v'),
//         fps, frameSize);
//     if (!writer.isOpened()) {
//         std::cerr << "Error: cannot open MP4 writer\n";
//         return -1;
//     }
//     for (auto& f : frames) writer << f;
//     writer.release();
//     std::cout << "Video saved as simulation.mp4" << std::endl;
//
//     // Final composite image
//     auto toColor=[&](const cv::Mat& M,int cmap){cv::Mat tmp,out;double mn,mx;cv::minMaxLoc(M,&mn,&mx);M.convertTo(tmp,CV_8U,255.0/(mx-mn),-255.0*mn/(mx-mn));cv::applyColorMap(tmp,out,cmap);return out;};
//     cv::Mat imgPsi=toColor(psi,cv::COLORMAP_OCEAN);
//     cv::Mat imgXi=toColor(xi,cv::COLORMAP_BONE);
//     cv::Mat imgTheta=toColor(Theta,cv::COLORMAP_INFERNO);
//     cv::Mat imgPhi=toColor(phi,cv::COLORMAP_HSV);
//     cv::Mat top,bottom,finalImg;
//     cv::hconcat(imgPsi,imgXi,top);
//     cv::hconcat(imgTheta,imgPhi,bottom);
//     cv::vconcat(top,bottom,finalImg);
//     cv::Mat finalBig;
//     cv::resize(finalImg, finalBig, cv::Size(finalImg.cols*scale, finalImg.rows*scale),0,0,cv::INTER_LANCZOS4);
//     cv::rotate(finalBig, finalBig, cv::ROTATE_180);
//     cv::imwrite("final.png", finalBig);
//     std::cout << "Final image saved as final.png" << std::endl;
//
//     return 0;
// }
