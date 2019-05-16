//
// Created by kv on 16/5/19.
//

#include "svo/handler.h"

namespace dr3 {

llui HandlerMono::_idx = 0;

HandlerBase::HandlerBase() : _state(State::FIRST_FRAME) {
    _monitor = new Monitor();
}

HandlerBase::~HandlerBase() =default;

HandlerMono::HandlerMono(dr3::AbstractCamera *cam)
    : HandlerBase(), _cam(cam), _initializer(nullptr) {
    initialize();

    // Add timers for various tasks
    _monitor->add_timer("pyramid");
    _monitor->add_timer("sparse_img_align");
    _monitor->add_timer("feature_align");
    _monitor->add_timer("pose_optimizer");
    _monitor->add_timer("local_BA");
}

HandlerMono::~HandlerMono() =default;

void HandlerMono::add_image(const cv::Mat &img, double timestamp) {
    _monitor->tic("pyramid");
    _frame_new.reset(new Frame(_idx++, img, _cam, timestamp));
    _monitor->toc("pyramid");

    if (_state == State::FIRST_FRAME) {
        process_first_frame();
    } else if (_state == State::SECOND_FRAME) {
        process_second_frame();
    } else if (_state == State::GENERAL_FRAME) {
        process_frame();
    } else {
        LOG(WARNING) << "Not handled case";
    }

    _frame_last = _frame_new;
    _frame_new.reset();
}

void HandlerMono::initialize() {

}

KState HandlerMono::process_first_frame() {
    _frame_new->_T_f_w = SE3(Matrix3d::Identity(), Vector3d::Zero());
    _initializer = new init::Init();
    if (_initializer->process_first_frame(_frame_new) == init::Result::FAILED) {
        return KState::KEYFRAME_NO;
    }
    _frame_new->set_keyframe();
    _map.add_frame(_frame_new);
    _state = State::SECOND_FRAME;
    LOG(INFO) << "Selected first keyframe";
    return KState::KEYFRAME_YES;
}

KState HandlerMono::process_second_frame() {
    init::Result res = _initializer->process_second_frame(_frame_new);
    if (res == init::Result::FAILED) {
        LOG(WARNING) << "Couldn't select the second frame";
        return KState::KEYFRAME_NO;
    }
    _frame_new->set_keyframe();
    _state = State::GENERAL_FRAME;
    _map.add_frame(_frame_new);
    LOG(INFO) << "Selected second keyframe and generated initial map!";
    return KState::KEYFRAME_YES;
}

KState HandlerMono::process_frame() {

}

} // namespace dr3
