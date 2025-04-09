import os
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm


class DetectionMetrics:
    def __init__(self, iou_thresholds=None, class_names=None):
        """
        初始化检测指标评估类
        
        参数:
            iou_thresholds: IoU阈值列表，如果为None则使用[0.5]
            class_names: 类别名称列表
        """
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else [0.5]
        self.class_names = class_names if class_names is not None else ["target"]
        self.num_classes = len(self.class_names)
        self.reset()
    
    def reset(self):
        """重置所有统计数据"""
        self.gt_counter_per_class = {i: 0 for i in range(self.num_classes)}
        self.detection_results = []
        self.frame_times = []
        self.frame_metrics = []  # 用于存储每帧的详细指标
    
    def compute_iou(self, box1, box2):
        """
        计算两个边界框的IoU
        
        参数:
            box1: [x1, y1, x2, y2] 格式的边界框
            box2: [x1, y1, x2, y2] 格式的边界框
            
        返回:
            iou: 两个边界框的IoU
        """
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 计算交集面积
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算并集面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        # 计算IoU
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou
    
    def convert_yolo_format(self, yolo_box, img_width, img_height):
        """
        将YOLO格式的边界框 [center_x, center_y, width, height] 转换为 [x1, y1, x2, y2] 格式
        
        参数:
            yolo_box: YOLO格式的边界框 [center_x, center_y, width, height]（归一化坐标）
            img_width: 图像宽度
            img_height: 图像高度
            
        返回:
            [x1, y1, x2, y2]: 边界框坐标（像素坐标）
        """
        center_x, center_y, width, height = yolo_box
        
        # 转换为像素坐标
        center_x *= img_width
        center_y *= img_height
        width *= img_width
        height *= img_height
        
        # 转换为 [x1, y1, x2, y2] 格式
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        
        return [x1, y1, x2, y2]

    def process_ground_truth(self, gt_file, img_width, img_height):
        """
        处理真实标签文件
        
        参数:
            gt_file: 真实标签文件路径
            img_width: 图像宽度
            img_height: 图像高度
            
        返回:
            ground_truths: 处理后的真实标签列表，每个元素为 [class_id, x1, y1, x2, y2]
        """
        ground_truths = []
        
        try:
            with open(gt_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip().split()
                if len(line) >= 5:  # 确保至少有类别ID和边界框坐标
                    class_id = int(line[0])
                    yolo_box = [float(x) for x in line[1:5]]
                    box = self.convert_yolo_format(yolo_box, img_width, img_height)
                    
                    ground_truths.append([class_id, *box])
                    self.gt_counter_per_class[class_id] += 1
        except Exception as e:
            print(f"处理真实标签文件 {gt_file} 时出错: {e}")
        
        return ground_truths
    
    def process_detections(self, detections):
        """
        处理检测结果
        
        参数:
            detections: 从模型推理得到的检测结果
            
        返回:
            processed_detections: 处理后的检测结果列表，每个元素为 [class_id, confidence, x1, y1, x2, y2]
        """
        processed_detections = []
        
        try:
            # 从supervision Detections对象转换
            if hasattr(detections, 'xyxy') and hasattr(detections, 'class_id') and hasattr(detections, 'confidence'):
                for i in range(len(detections.xyxy)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    class_id = detections.class_id[i]
                    confidence = detections.confidence[i]
                    processed_detections.append([class_id, confidence, x1, y1, x2, y2])
            # 从ultralytics结果转换
            elif hasattr(detections, 'boxes') and hasattr(detections.boxes, 'xyxy'):
                boxes = detections.boxes.xyxy.cpu().numpy()
                class_ids = detections.boxes.cls.cpu().numpy()
                confidences = detections.boxes.conf.cpu().numpy()
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    class_id = int(class_ids[i])
                    confidence = confidences[i]
                    processed_detections.append([class_id, confidence, x1, y1, x2, y2])
        except Exception as e:
            print(f"处理检测结果时出错: {e}")
        
        return processed_detections
    
    def evaluate_image(self, gt_file, detections, img_width, img_height, frame_idx=None):
        """
        评估单张图像的检测结果
        
        参数:
            gt_file: 真实标签文件路径
            detections: 检测结果
            img_width: 图像宽度
            img_height: 图像高度
            frame_idx: 帧索引，用于输出
        """
        # 记录开始时间
        start_time = time.time()
        
        # 处理真实标签
        ground_truths = self.process_ground_truth(gt_file, img_width, img_height)
        
        # 处理检测结果
        processed_detections = self.process_detections(detections)
        
        # 记录处理结果
        for det in processed_detections:
            self.detection_results.append({
                'class_id': det[0],
                'confidence': det[1],
                'bbox': det[2:],
                'gt_file': gt_file,
                'ground_truths': ground_truths
            })
        
        # 计算处理时间
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        
        # 计算当前帧的TP、FP、FN
        frame_metrics = {'frame_time': frame_time}
        frame_tp, frame_fp, frame_fn = 0, 0, 0
        
        # 为每个类别计算单帧指标
        for iou_threshold in self.iou_thresholds:
            for class_id in range(self.num_classes):
                # 筛选当前帧和类别的检测结果
                class_dets = [d for d in processed_detections if d[0] == class_id]
                class_gts = [gt for gt in ground_truths if gt[0] == class_id]
                
                # 记录已匹配的真实标签
                matched_gt_indices = []
                
                # 计算TP和FP
                for det in class_dets:
                    _, conf, *bbox = det
                    max_iou = -np.inf
                    max_gt_idx = -1
                    
                    # 找到最大IoU的真实标签
                    for j, gt in enumerate(class_gts):
                        if j in matched_gt_indices:
                            continue
                        
                        _, gt_x1, gt_y1, gt_x2, gt_y2 = gt
                        gt_bbox = [gt_x1, gt_y1, gt_x2, gt_y2]
                        iou = self.compute_iou(bbox, gt_bbox)
                        
                        if iou > max_iou:
                            max_iou = iou
                            max_gt_idx = j
                    
                    # 判断是否为TP或FP
                    if max_iou >= iou_threshold and max_gt_idx not in matched_gt_indices:
                        frame_tp += 1
                        matched_gt_indices.append(max_gt_idx)
                    else:
                        frame_fp += 1
                
                # 计算FN
                frame_fn += len(class_gts) - len(matched_gt_indices)
        
        frame_metrics['tp'] = frame_tp
        frame_metrics['fp'] = frame_fp
        frame_metrics['fn'] = frame_fn
        frame_metrics['iou_thresholds'] = self.iou_thresholds
        
        # 存储并输出每帧指标
        self.frame_metrics.append(frame_metrics)
        
        # 输出每帧计算结果
        frame_label = f"Frame {frame_idx}" if frame_idx is not None else "Current frame"
        print(f"\n{frame_label} 指标:")
        print(f"  处理时间: {frame_time*1000:.2f} ms")
        #print(f"  IoU阈值: {self.iou_thresholds}")
        
        return frame_metrics
    
    def compute_metrics(self, confidence_thresholds=None):
        """
        计算所有评估指标
        
        参数:
            confidence_thresholds: 置信度阈值列表，用于计算不同置信度下的PR值
            
        返回:
            metrics: 包含所有评估指标的字典
        """
        results = {}
        
        # 使用默认置信度阈值如果未提供
        if confidence_thresholds is None:
            confidence_thresholds = np.linspace(0.1, 0.9, 9)  # 默认: 0.1到0.9之间的9个值
        
        # 计算mAP@50
        results['mAP50'] = self.compute_map(iou_threshold=0.5)
        
        # 计算mAP@50-95
        ap_values = []
        for iou_threshold in np.arange(0.5, 1.0, 0.05):
            ap_values.append(self.compute_map(iou_threshold=iou_threshold))
        results['mAP50-95'] = np.mean(ap_values)
        
        # 计算平均处理时间
        results['avg_frame_time'] = np.mean(self.frame_times) if self.frame_times else 0
        results['fps'] = 1.0 / results['avg_frame_time'] if results['avg_frame_time'] > 0 else 0
        
        # 计算总体TP、FP、FN
        total_tp = sum(frame['tp'] for frame in self.frame_metrics)
        total_fp = sum(frame['fp'] for frame in self.frame_metrics)
        total_fn = sum(frame['fn'] for frame in self.frame_metrics)
        
        results['total_tp'] = total_tp
        results['total_fp'] = total_fp
        results['total_fn'] = total_fn
        
        # 计算总体precision和recall
        results['precision'] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        results['recall'] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        # 计算每个类别的指标
        results['per_class'] = {}
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            results['per_class'][class_name] = {
                'AP50': self.compute_ap_per_class(class_id, iou_threshold=0.5),
                'gt_count': self.gt_counter_per_class[class_id]
            }
            
            # 计算各置信度下的精度和召回率
            precision_recall = self.compute_precision_recall_curve(class_id, iou_threshold=0.5)
            results['per_class'][class_name]['precision_recall'] = precision_recall
            
            # 对不同置信度阈值计算Precision和Recall
            thresholds_data = {}
            class_detections = [d for d in self.detection_results if d['class_id'] == class_id]
            
            for conf_threshold in confidence_thresholds:
                # 筛选符合置信度阈值的检测结果
                filtered_detections = [d for d in class_detections if d['confidence'] >= conf_threshold]
                
                # 统计GT数量
                gt_count = self.gt_counter_per_class[class_id]
                
                # 初始化TP、FP计数
                tp_count = 0
                fp_count = 0
                
                # 用于记录已匹配的真实标签
                gt_matched = defaultdict(list)
                
                # 评估每个检测结果
                for detection in filtered_detections:
                    bbox = detection['bbox']
                    gt_file = detection['gt_file']
                    ground_truths = detection['ground_truths']
                    
                    # 筛选出当前类别的真实标签
                    gt_class = [gt for gt in ground_truths if gt[0] == class_id]
                    
                    # 记录最大IoU和对应的真实标签索引
                    max_iou = -np.inf
                    max_gt_idx = -1
                    
                    for j, gt in enumerate(gt_class):
                        _, gt_x1, gt_y1, gt_x2, gt_y2 = gt
                        gt_bbox = [gt_x1, gt_y1, gt_x2, gt_y2]
                        
                        # 计算IoU
                        iou = self.compute_iou(bbox, gt_bbox)
                        
                        if iou > max_iou:
                            max_iou = iou
                            max_gt_idx = j
                    
                    # 判断是否为TP或FP
                    if max_iou >= 0.5:  # 使用标准IoU阈值0.5
                        # 检查该真实标签是否已被匹配
                        if max_gt_idx not in gt_matched[gt_file]:
                            tp_count += 1  # TP
                            gt_matched[gt_file].append(max_gt_idx)
                        else:
                            fp_count += 1  # FP（重复检测）
                    else:
                        fp_count += 1  # FP（无匹配或IoU不足）
                
                # 计算FN
                fn_count = gt_count - tp_count
                
                # 计算precision和recall
                precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
                recall = tp_count / gt_count if gt_count > 0 else 0
                
                thresholds_data[conf_threshold] = {
                    'precision': precision,
                    'recall': recall,
                    'tp': tp_count,
                    'fp': fp_count,
                    'fn': fn_count
                }
            
            # 添加到结果中
            results['per_class'][class_name]['thresholds'] = thresholds_data
        
        return results
    
    def compute_ap_per_class(self, class_id, iou_threshold=0.5):
        """
        计算每个类别的平均精度 (AP)
        
        参数:
            class_id: 类别ID
            iou_threshold: IoU阈值
            
        返回:
            ap: 类别的平均精度
        """
        precision_recall = self.compute_precision_recall_curve(class_id, iou_threshold)
        if not precision_recall or 'precision' not in precision_recall or 'recall' not in precision_recall:
            return 0.0
        
        # 计算AP（使用所有点的平均值）
        precision = precision_recall['precision']
        recall = precision_recall['recall']
        
        # 使用11点插值方法计算AP（VOC2007方法）
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
            
        return ap
    
    def compute_map(self, iou_threshold=0.5):
        """
        计算所有类别的平均精度的平均值 (mAP)
        
        参数:
            iou_threshold: IoU阈值
            
        返回:
            mAP: 所有类别的平均精度的平均值
        """
        ap_values = []
        
        for class_id in range(self.num_classes):
            if self.gt_counter_per_class[class_id] > 0:
                ap = self.compute_ap_per_class(class_id, iou_threshold)
                ap_values.append(ap)
        
        mAP = np.mean(ap_values) if ap_values else 0.0
        return mAP
    
    def compute_precision_recall_curve(self, class_id, iou_threshold=0.5):
        """
        计算精度-召回率曲线
        
        参数:
            class_id: 类别ID
            iou_threshold: IoU阈值
            
        返回:
            precision_recall: 包含精度和召回率的字典
        """
        # 筛选指定类别的检测结果
        class_detections = [d for d in self.detection_results if d['class_id'] == class_id]
        
        # 按置信度降序排序
        class_detections = sorted(class_detections, key=lambda x: x['confidence'], reverse=True)
        
        # 统计该类别的真实标签数量
        gt_count = self.gt_counter_per_class[class_id]
        if gt_count == 0:
            return {'precision': np.array([]), 'recall': np.array([]), 'tp': np.array([]), 'fp': np.array([])}
        
        # 初始化TP、FP计数器
        tp = np.zeros(len(class_detections))
        fp = np.zeros(len(class_detections))
        
        # 用于记录已匹配的真实标签
        gt_matched = defaultdict(list)
        
        # 遍历所有检测结果
        for i, detection in enumerate(class_detections):
            bbox = detection['bbox']
            gt_file = detection['gt_file']
            ground_truths = detection['ground_truths']
            
            # 筛选出当前类别的真实标签
            gt_class = [gt for gt in ground_truths if gt[0] == class_id]
            
            # 记录最大IoU和对应的真实标签索引
            max_iou = -np.inf
            max_gt_idx = -1
            
            for j, gt in enumerate(gt_class):
                _, gt_x1, gt_y1, gt_x2, gt_y2 = gt
                gt_bbox = [gt_x1, gt_y1, gt_x2, gt_y2]
                
                # 计算IoU
                iou = self.compute_iou(bbox, gt_bbox)
                
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = j
            
            # 判断是否为TP或FP
            if max_iou >= iou_threshold:
                # 检查该真实标签是否已被匹配
                if max_gt_idx not in gt_matched[gt_file]:
                    tp[i] = 1  # TP
                    gt_matched[gt_file].append(max_gt_idx)
                else:
                    fp[i] = 1  # FP（重复检测）
            else:
                fp[i] = 1  # FP（无匹配或IoU不足）
        
        # 计算累积TP和FP
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        
        # 计算精度和召回率
        precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)
        recall = cumsum_tp / gt_count
        
        # 返回精度、召回率、TP和FP
        return {
            'precision': precision,
            'recall': recall,
            'tp': cumsum_tp,
            'fp': cumsum_fp,
            'fn': gt_count - cumsum_tp[-1] if len(cumsum_tp) > 0 else gt_count
        }
    
    def plot_precision_recall_curves(self, save_path=None):
        """
        绘制所有类别的精度-召回率曲线
        
        参数:
            save_path: 保存图像的路径，如果为None则显示图像
        """
        plt.figure(figsize=(10, 8))
        
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            precision_recall = self.compute_precision_recall_curve(class_id, iou_threshold=0.5)
            
            if 'precision' in precision_recall and 'recall' in precision_recall:
                precision = precision_recall['precision']
                recall = precision_recall['recall']
                
                if len(precision) > 0 and len(recall) > 0:
                    ap = self.compute_ap_per_class(class_id, iou_threshold=0.5)
                    plt.plot(recall, precision, label=f'{class_name} (AP: {ap:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def print_results(self, metrics):
        """
        打印评估结果
        
        参数:
            metrics: compute_metrics()函数的返回结果
        """
        print("\n" + "="*50)
        print("检测指标评估结果")
        print("="*50)
        
        print(f"\n平均每帧计算耗时: {metrics['avg_frame_time']*1000:.2f} ms")
        print(f"FPS: {metrics['fps']:.2f}")
        
        print(f"\n总体统计:")
        print(f"TP: {metrics['total_tp']}")
        print(f"FP: {metrics['total_fp']}")
        print(f"FN: {metrics['total_fn']}")
        
        print(f"\nmAP@50: {metrics['mAP50']:.4f}")
        print(f"mAP@50-95: {metrics['mAP50-95']:.4f}")
        
        print("\n类别详细指标:")
        print("-"*90)
        print(f"{'类别':<10} {'AP@50':<8} {'GT数量':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10}")
        print("-"*90)
        
        for class_name, class_metrics in metrics['per_class'].items():
            ap50 = class_metrics['AP50']
            gt_count = class_metrics['gt_count']
            
            pr_data = class_metrics.get('precision_recall', {})
            tp = pr_data.get('tp', [0])[-1] if 'tp' in pr_data and len(pr_data['tp']) > 0 else 0
            fp = pr_data.get('fp', [0])[-1] if 'fp' in pr_data and len(pr_data['fp']) > 0 else 0
            fn = pr_data.get('fn', 0)
            
            # 计算最终的Precision和Recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"{class_name:<10} {ap50:<8.4f} {gt_count:<8} {int(tp):<6} {int(fp):<6} {int(fn):<6} {precision:<10.4f} {recall:<10.4f}")
        
        print("="*90)
        
        # 输出P-R曲线关键点数据
        print("\nPrecision-Recall曲线数据:")
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            pr_data = metrics['per_class'][class_name].get('precision_recall', {})
            
            if 'precision' in pr_data and 'recall' in pr_data:
                precision = pr_data['precision']
                recall = pr_data['recall']
                
                if len(precision) > 0 and len(recall) > 0:
                    # 选择最多10个关键点输出
                    points_count = min(10, len(precision))
                    indices = np.linspace(0, len(precision)-1, points_count, dtype=int)
                    
                    print(f"\n{class_name} 类别的P-R曲线关键点:")
                    print(f"{'Recall':<10} {'Precision':<10}")
                    print("-"*25)
                    
                    for idx in indices:
                        print(f"{recall[idx]:<10.4f} {precision[idx]:<10.4f}")
        
        print("="*50)


def evaluate_model(model, dataset, gt_dir, class_names=None, output_dir=None, predict_fn=None, iou_thresholds=None, confidence_thresholds=None):
    """
    评估模型性能
    
    参数:
        model: YOLO模型
        dataset: 数据集
        gt_dir: 包含真实标签的目录
        class_names: 类别名称列表
        output_dir: 输出目录
        predict_fn: 预测函数
        iou_thresholds: IoU阈值列表
        confidence_thresholds: 置信度阈值列表，用于计算不同置信度下的PR值
    
    返回:
        metrics: 评估指标
    """
    import cv2  # 导入cv2用于读取图像
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化指标评估类
    metrics = DetectionMetrics(iou_thresholds=iou_thresholds, class_names=class_names)
    
    # 获取第一帧以确定图像尺寸
    first_frame = dataset[0]
    if isinstance(first_frame, tuple):
        first_frame = first_frame[0]  # 假设第一个元素是WL图像
    
    if isinstance(first_frame, str):
        img = cv2.imread(first_frame)
        height, width = img.shape[:2]
    else:
        height, width = first_frame.shape[:2]
    
    # 评估每一帧
    for i, data in enumerate(tqdm(dataset, desc="评估进度")):
        # 提取图像文件名
        if isinstance(data, tuple):
            wl_frame, ir_frame = data
            if isinstance(wl_frame, str):
                # 如果是文件路径，提取文件名
                filename = os.path.basename(wl_frame)
                filename = os.path.splitext(filename)[0]
                if filename.startswith("wl_"):
                    filename = filename[3:]
            else:
                # 如果是图像数据，使用索引作为文件名
                filename = f"{i:04d}"
        else:
            # 单图像情况
            if isinstance(data, str):
                filename = os.path.basename(data)
                filename = os.path.splitext(filename)[0]
            else:
                filename = f"{i:04d}"
        
        # 构建对应的真实标签文件路径
        gt_file = os.path.join(gt_dir, f"wuxi_2_{filename}.txt")
        if not os.path.exists(gt_file):
            print(f"警告: 真实标签文件 {gt_file} 不存在")
            continue
        
        # 开始计时
        start_time = time.time()
        
        # 进行预测
        if isinstance(data, tuple):
            results = predict_fn([data[0], data[1]])  # 使用传入的predict_fn
        else:
            results = predict_fn(data)  # 使用传入的predict_fn

        if isinstance(results, list):
            result = results[0]
        else:
            result = next(results)
        
        # 结束计时
        inference_time = time.time() - start_time
        
        # 输出每帧计算耗时
        print(f"\n帧 {i}: 推理耗时 = {inference_time*1000:.2f} ms")
        
        # 评估结果并输出帧统计信息
        frame_metrics = metrics.evaluate_image(gt_file, result, width, height, frame_idx=i)
    
    # 计算整体指标（使用传入的置信度阈值）
    results = metrics.compute_metrics(confidence_thresholds=confidence_thresholds)
    
    # 打印汇总结果
    metrics.print_results(results)
    
    # 绘制PR曲线
    if output_dir:
        metrics.plot_precision_recall_curves(save_path=os.path.join(output_dir, 'pr_curves.png'))
        
        # 保存不同置信度阈值下的PR值到CSV文件
        if confidence_thresholds:
            import pandas as pd
            
            # 为每个类别创建PR曲线数据
            for class_id in range(metrics.num_classes):
                class_name = metrics.class_names[class_id]
                class_data = results['per_class'][class_name]
                
                if 'thresholds' in class_data:
                    # 准备数据
                    conf_values = []
                    precision_values = []
                    recall_values = []
                    
                    for conf, pr_data in class_data['thresholds'].items():
                        conf_values.append(conf)
                        precision_values.append(pr_data['precision'])
                        recall_values.append(pr_data['recall'])
                    
                    # 创建DataFrame
                    pr_df = pd.DataFrame({
                        'confidence': conf_values,
                        'precision': precision_values,
                        'recall': recall_values
                    })
                    
                    # 保存到CSV
                    pr_df.to_csv(os.path.join(output_dir, f'{class_name}_pr_curve.csv'), index=False)
    
    return results


# 使用示例
if __name__ == "__main__":
    import cv2
    from your_model_import import predict  # 替换为你的实际导入语句
    from your_dataset_import import get_dataset  # 替换为你的实际导入语句
    
    # 定义类别名称
    class_names = ["person", "bicycle", "car"]  # 替换为你的实际类别
    # 定义IoU阈值
    iou_thresholds = [0.5, 0.75]  # 可以设置多个IoU阈值
    # 获取数据集
    dataset = get_dataset('path/to/dataset')
    
    # 定义置信度阈值列表，用于计算PR曲线
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 评估模型
    results = evaluate_model(
        model=None,  # 不需要传入模型，因为使用了predict_fn函数
        dataset=dataset,
        gt_dir='path/to/ground_truth',
        class_names=class_names,
        output_dir='path/to/output',
        predict_fn=predict,  # 传入预测函数
        iou_thresholds=iou_thresholds,
        confidence_thresholds=confidence_thresholds
    )

