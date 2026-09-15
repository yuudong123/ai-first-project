/* 설비가 누락되면 다른 설비 값으로 대체하지 않는다. */
(function(root){
  function select(payload,id){
    if (Array.isArray(payload.equipment_states)){
      const state = payload.equipment_states.find(s => s.equipment_id === id);
      return state || {equipment_id:id,event_id:0,elapsed_sec:0,run_id:payload.run_id,sensors:{},
        prediction:{status:'waiting',observed_window_sec:0,components:{}}};
    }
    return {equipment_id:id,event_id:0,elapsed_sec:0,run_id:payload?.run_id,sensors:{},
      prediction:{status:'waiting',observed_window_sec:0,components:{}}};
  }
  if(typeof module !== 'undefined' && module.exports) module.exports={select};
  else root.EquipmentState={select};
})(globalThis);
