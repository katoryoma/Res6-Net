"""
Celeb-DFデータセット構造の検証・確認スクリプト

データセットの整合性をチェックし、各動画のフレーム数などの情報を表示します。
"""

import os
from collections import defaultdict
from utils.config import CELEBDF_ROOT, CELEBDF_VAL_SPLIT, CROSS_DATASET_TEST_DATASETS


def load_balanced_synthesis_list():
    """
    balanced_synthesis.txt から固定の synthesis 動画 ID リストを読み込む
    
    Returns:
        set: balanced_synthesis.txt に記載された動画 ID のセット
    """
    balanced_list_path = "balanced_synthesis.txt"
    
    if not os.path.exists(balanced_list_path):
        print(f"⚠ Warning: {balanced_list_path} not found. Using all synthesis videos.")
        return None
    
    video_ids = set()
    try:
        with open(balanced_list_path, "r") as f:
            for line in f:
                video_id = line.strip()
                if video_id:
                    video_ids.add(video_id)
        print(f"✓ Loaded {len(video_ids)} fixed synthesis videos from balanced_synthesis.txt")
    except Exception as e:
        print(f"Error reading {balanced_list_path}: {e}")
    
    return video_ids if video_ids else None


def inspect_dataset(class_dir, class_name="Dataset", filter_videos=None):
    """
    データセット構造を検証・表示
    
    Args:
        class_dir (str): クラスディレクトリ（<videoID>を含むディレクトリ）
        class_name (str): クラスの名前（"Real" または "Synthesis"）
        filter_videos (set, optional): 対象とするビデオIDのセット（Noneの場合は全て対象）
    """
    print(f"\n{'='*80}")
    print(f"Inspecting {class_name}")
    print(f"Directory: {class_dir}")
    if filter_videos is not None:
        print(f"Using filtered video list (total: {len(filter_videos)} videos)")
    print(f"{'='*80}\n")
    
    if not os.path.exists(class_dir):
        print(f"❌ Directory does not exist: {class_dir}")
        return None
    
    videos = []
    video_frame_counts = []
    total_frames = 0
    filtered_count = 0
    
    for video_id in sorted(os.listdir(class_dir)):
        video_path = os.path.join(class_dir, video_id)
        
        if not os.path.isdir(video_path):
            continue
        
        # filter_videos が指定されている場合は、そのセットに含まれるビデオのみを対象
        if filter_videos is not None and video_id not in filter_videos:
            filtered_count += 1
            continue
        
        # フレームファイルを探す
        frames = [f for f in os.listdir(video_path) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
        frame_count = len(frames)
        
        if frame_count > 0:
            videos.append({
                "video_id": video_id,
                "frame_count": frame_count
            })
            video_frame_counts.append(frame_count)
            total_frames += frame_count
    
    video_count = len(videos)
    avg_frames = sum(video_frame_counts) / len(video_frame_counts) if video_frame_counts else 0
    min_frames = min(video_frame_counts) if video_frame_counts else 0
    max_frames = max(video_frame_counts) if video_frame_counts else 0
    
    # 結果表示
    print(f"Videos:           {video_count}")
    if filter_videos is not None:
        print(f"Filtered out:     {filtered_count} videos")
    print(f"Total Frames:     {total_frames}")
    print(f"Avg Frames/Video: {avg_frames:.1f}")
    print(f"Min Frames:       {min_frames}")
    print(f"Max Frames:       {max_frames}")
    
    # Show sample videos
    print(f"\nSample Videos (first 5):")
    for video in videos[:5]:
        print(f"  - {video['video_id']}: {video['frame_count']} frames")
    if len(videos) > 5:
        print(f"  ... and {len(videos) - 5} more")
    
    print(f"\n{'='*80}\n")
    
    return {
        "video_count": video_count,
        "frame_count": total_frames,
        "avg_frames_per_video": avg_frames,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "videos": videos
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Celeb-DF Dataset Inspection")
    print("="*80)
    
    # balanced_synthesis.txt を読み込む（trainデータのフィルタリング用）
    balanced_synthesis_videos = load_balanced_synthesis_list()
    
    # 各データセットを検査
    real_train = inspect_dataset(
        os.path.join(CELEBDF_ROOT, "real", "train"),
        "Real Train Dataset"
    )
    
    # synthesis train は balanced_synthesis.txt でフィルタリング
    syn_train = inspect_dataset(
        os.path.join(CELEBDF_ROOT, "synthesis", "train"),
        "Synthesis Train Dataset",
        filter_videos=balanced_synthesis_videos
    )
    
    # テストセットはフィルタリングなし
    real_test = inspect_dataset(
        os.path.join(CELEBDF_ROOT, "real", "test"),
        "Real Test Dataset"
    )
    
    syn_test = inspect_dataset(
        os.path.join(CELEBDF_ROOT, "synthesis", "test"),
        "Synthesis Test Dataset"
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if real_train and syn_train:
        print("\n[Train Dataset]")
        print(f"  Real:      {real_train['video_count']:3} videos, {real_train['frame_count']:6} frames")
        print(f"  Synthesis: {syn_train['video_count']:3} videos, {syn_train['frame_count']:6} frames")
        
        # Balance check
        if syn_train["video_count"] > real_train["video_count"]:
            print(f"  ✓ Synthesis has {syn_train['video_count'] - real_train['video_count']} more videos")
            print(f"    → Will be sampled to match Real during training")
        elif syn_train["video_count"] == real_train["video_count"]:
            print(f"  ✓ Balanced: Real and Synthesis have same video count")
        else:
            print(f"  ⚠ Synthesis has fewer videos than Real")
        
        # 総フレーム数の計算
        train_total_frames = real_train['frame_count'] + syn_train['frame_count']
        print(f"\n  Total Train Frames: {train_total_frames}")
        
        # Cross-Validation split
        print(f"\n[Cross-Validation Split (train:val = {100*(1-CELEBDF_VAL_SPLIT):.0f}:{100*CELEBDF_VAL_SPLIT:.0f})]")
        print(f"  Real Train: {int(real_train['video_count'] * (1 - CELEBDF_VAL_SPLIT)):3} videos, {int(real_train['frame_count'] * (1 - CELEBDF_VAL_SPLIT)):6} frames")
        print(f"  Real Val:   {int(real_train['video_count'] * CELEBDF_VAL_SPLIT):3} videos, {int(real_train['frame_count'] * CELEBDF_VAL_SPLIT):6} frames")
        print(f"  Syn Train:  {int(syn_train['video_count'] * (1 - CELEBDF_VAL_SPLIT)):3} videos, {int(syn_train['frame_count'] * (1 - CELEBDF_VAL_SPLIT)):6} frames")
        print(f"  Syn Val:    {int(syn_train['video_count'] * CELEBDF_VAL_SPLIT):3} videos, {int(syn_train['frame_count'] * CELEBDF_VAL_SPLIT):6} frames")
        
        # 訓練時の総フレーム数
        train_real_frames = int(real_train['frame_count'] * (1 - CELEBDF_VAL_SPLIT))
        train_syn_frames = int(syn_train['frame_count'] * (1 - CELEBDF_VAL_SPLIT))
        train_total_cv = train_real_frames + train_syn_frames
        print(f"\n  Total Train Frames (CV split): {train_total_cv}")
    
    if real_test and syn_test:
        print("\n[Test Dataset]")
        print(f"  Real:      {real_test['video_count']:3} videos, {real_test['frame_count']:6} frames")
        print(f"  Synthesis: {syn_test['video_count']:3} videos, {syn_test['frame_count']:6} frames")
        test_total_frames = real_test['frame_count'] + syn_test['frame_count']
        print(f"  Total Test Frames: {test_total_frames}")
    
    print("\n" + "="*80)
    print("✓ Dataset inspection completed!")
    print("="*80 + "\n")
