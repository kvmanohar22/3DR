//
// Created by kv on 16/5/19.
//

#ifndef INC_3DR_HANDLER_H
#define INC_3DR_HANDLER_H

#include "global.hpp"
#include "config.hpp"
#include "frame.hpp"
#include "point.hpp"
#include "camera.hpp"
#include "timer.hpp"
#include "initialization.hpp"

namespace dr3 {

enum class State {
    FIRST_FRAME,
    SECOND_FRAME,
    GENERAL_FRAME
};

enum class KState {
    KEYFRAME_YES,
    KEYFRAME_NO
};

class HandlerBase {
public:
    State           _state;       /// Current state of the system
    Map             _map;         /// Set of keyframes & points
    Monitor        *_monitor;     /// Monitor state of the system

    explicit HandlerBase();
    virtual ~HandlerBase();

    /// Get the current map.
    inline const Map& map() const { return _map; }

    /// Get the current state of the algorithm.
    inline State stage() const { return _state; }

}; // class HandlerBase

class HandlerMono : public HandlerBase {
public:
    AbstractCamera *_cam;         /// camera model
    FramePtr        _frame_last;  /// last frame
    FramePtr        _frame_new;   /// new frame
    init::Init     *_initializer; /// Initial map estimation
    static llui     _idx;         /// unique index of frame

    explicit HandlerMono(AbstractCamera* cam);
    ~HandlerMono() override;

    /// Provide an image.
    void add_image(const cv::Mat& img, double timestamp);

    /// Get the last frame that has been processed.
    FramePtr last_frame() { return _frame_last; }

    /// Initialize the visual odometry algorithm.
    virtual void initialize();

    /// Processes the first frame and sets it as a keyframe.
    virtual KState process_first_frame();

    /// Processes all frames after the first frame until a keyframe is selected.
    virtual KState process_second_frame();

    /// Processes all frames after the first two keyframes.
    virtual KState process_frame();

}; // class HandlerMono

} // namespace dr3

#endif //INC_3DR_HANDLER_H
