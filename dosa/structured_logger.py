import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class StructuredLogger:
    """
    统一的结构化日志器，支持控制台输出和文件记录
    
    功能：
    - 控制台仅输出最少必要信息
    - 详细信息写入文件
    - 支持事件、试验、工件记录
    """
    
    def __init__(self, 
                 log_dir: str = "output",
                 run_id: Optional[str] = None,
                 minimal_console: bool = True,
                 log_intermediate: bool = True):
        """
        初始化结构化日志器
        
        Args:
            log_dir: 日志输出目录
            run_id: 运行ID，如果为None则自动生成
            minimal_console: 是否最小化控制台输出
            log_intermediate: 是否记录中间过程
        """
        self.log_dir = Path(log_dir)
        self.run_id = run_id or self._generate_run_id()
        self.minimal_console = minimal_console
        self.log_intermediate = log_intermediate
        
        # 创建运行目录
        self.run_dir = self.log_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.run_dir / "groups").mkdir(exist_ok=True)
        (self.run_dir / "artifacts").mkdir(exist_ok=True)
        
        # 初始化日志文件
        self.events_file = self.run_dir / "events.jsonl"
        self.trials_file = self.run_dir / "trials.jsonl"
        self.meta_file = self.run_dir / "run_meta.json"
        
        # 记录运行开始时间
        self.start_time = time.time()
        
        # 写入运行元信息
        self._write_run_meta()
        
        # 记录运行开始事件
        self.event("run_start", timestamp=self.start_time)
    
    def _generate_run_id(self) -> str:
        """生成唯一的运行ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{short_uuid}"
    
    def _write_run_meta(self):
        """写入运行元信息"""
        meta = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "start_time_iso": datetime.fromtimestamp(self.start_time).isoformat(),
            "log_dir": str(self.run_dir),
            "minimal_console": self.minimal_console,
            "log_intermediate": self.log_intermediate,
            "version": "1.0"
        }
        
        with open(self.meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def event(self, name: str, **kwargs):
        """
        记录事件
        
        Args:
            name: 事件名称
            **kwargs: 事件相关数据
        """
        event_data = {
            "event": name,
            "timestamp": time.time(),
            **kwargs
        }
        
        # 写入事件文件
        if self.log_intermediate:
            with open(self.events_file, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
        
        # 控制台输出关键事件
        if name in ["run_start", "run_end", "outer_step_start", "phase_start", "new_best"]:
            if name == "run_start":
                self.console(f"=== DSE Run Started (ID: {self.run_id}) ===")
            elif name == "run_end":
                duration = kwargs.get('duration', 0)
                self.console(f"=== DSE Run Completed in {duration:.2f}s ===")
            elif name == "outer_step_start":
                step = kwargs.get('index', 0)
                total = kwargs.get('total', 0)
                self.console(f"\n--- Outer Step {step}/{total} ---")
            elif name == "phase_start":
                phase = kwargs.get('phase', '')
                self.console(f"--- Phase: {phase} ---")
            elif name == "new_best":
                metrics = kwargs.get('metrics', {})
                loss = metrics.get('loss', 0)
                edp = metrics.get('edp', 0)
                area = metrics.get('area_mm2', 0)
                self.console(f"🎯 New Best Found! Loss: {loss:.4f}, EDP: {edp:.2e}, Area: {area:.2f}mm²")
    
    def trial(self, step: int, payload: Dict[str, Any]):
        """
        记录试验结果
        
        Args:
            step: 试验步数
            payload: 试验数据（包含指标、参数等）
        """
        trial_data = {
            "step": step,
            "timestamp": time.time(),
            **payload
        }
        
        # 写入试验文件
        if self.log_intermediate:
            with open(self.trials_file, 'a') as f:
                f.write(json.dumps(trial_data) + '\n')
        
        # 控制台输出（仅在非最小模式或重要试验时）
        if not self.minimal_console or payload.get('best_so_far', False):
            metrics = payload.get('metrics', {})
            loss = metrics.get('loss', 0)
            edp = metrics.get('edp', 0)
            area = metrics.get('area_mm2', 0)
            
            if payload.get('best_so_far', False):
                self.console(f"  Step {step}: ⭐ BEST - Loss={loss:.4f}, EDP={edp:.2e}, Area={area:.2f}mm²")
            elif step % 50 == 0:  # 每50步输出一次
                self.console(f"  Step {step}: Loss={loss:.4f}, EDP={edp:.2e}, Area={area:.2f}mm²")
    
    def artifact(self, path: str, meta: Dict[str, Any]):
        """
        记录工件信息
        
        Args:
            path: 工件路径
            meta: 工件元信息
        """
        artifact_data = {
            "path": path,
            "timestamp": time.time(),
            "meta": meta
        }
        
        # 记录到事件日志
        self.event("artifact_saved", **artifact_data)
    
    def console(self, msg: str):
        """
        控制台输出
        
        Args:
            msg: 消息内容
        """
        print(msg)
    
    def group_stats(self, group_id: str, stats: Dict[str, Any]):
        """
        记录融合组统计信息
        
        Args:
            group_id: 组ID
            stats: 统计数据
        """
        if not self.log_intermediate:
            return
            
        group_file = self.run_dir / "groups" / f"{group_id}.json"
        
        stats_data = {
            "group_id": group_id,
            "timestamp": time.time(),
            "stats": stats
        }
        
        with open(group_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
    
    def finalize(self, final_results: Dict[str, Any]):
        """
        完成日志记录
        
        Args:
            final_results: 最终结果
        """
        end_time = time.time()
        duration = end_time - self.start_time
        
        # 更新运行元信息
        with open(self.meta_file, 'r') as f:
            meta = json.load(f)
        
        meta.update({
            "end_time": end_time,
            "end_time_iso": datetime.fromtimestamp(end_time).isoformat(),
            "duration": duration,
            "final_results": final_results
        })
        
        with open(self.meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        
        # 记录运行结束事件
        self.event("run_end", duration=duration, **final_results)
        
        # 控制台输出总结
        best_loss = final_results.get('best_loss', float('inf'))
        total_trials = final_results.get('total_trials', 0)
        
        if best_loss != float('inf'):
            best_metrics = final_results.get('best_metrics', {})
            edp = best_metrics.get('edp', 0)
            area = best_metrics.get('area_mm2', 0)
            
            self.console(f"\n📊 Final Results:")
            self.console(f"   Best Loss: {best_loss:.4f}")
            self.console(f"   Best EDP: {edp:.2e}")
            self.console(f"   Best Area: {area:.2f}mm²")
            self.console(f"   Total Trials: {total_trials}")
            self.console(f"   Duration: {duration:.2f}s")
        else:
            self.console(f"\n❌ No valid solutions found in {total_trials} trials")
        
        self.console(f"\n📁 Logs saved to: {self.run_dir}")
    
    def get_run_dir(self) -> Path:
        """获取运行目录路径"""
        return self.run_dir
    
    def get_run_id(self) -> str:
        """获取运行ID"""
        return self.run_id