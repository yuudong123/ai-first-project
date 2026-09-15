/* 기준 이탈 표시는 AI 분류와 독립적이며, 누락·구간 전환 시 연속 횟수를 초기화한다. */
(function(root){
  function outside(value, band){
    if (!Number.isFinite(value) || !band || !Number.isFinite(band.lower) || !Number.isFinite(band.upper)) return null;
    return value < band.lower || value > band.upper;
  }
  function advance(previous, data, bands, sensors){
    const stamp = Date.parse(data.generated_at || data.updated_at);
    const contiguous = previous && previous.run === data.run_id && previous.segment === data.segment_id
      && previous.event + 1 === data.event_id && stamp > previous.stamp && stamp-previous.stamp <= 2000;
    const counts = {};
    sensors.forEach(sensor => {
      counts[sensor] = outside(data.sensors?.[sensor],bands?.sensors?.[sensor]) === true
        ? (contiguous ? previous.counts[sensor] || 0 : 0)+1 : 0;
    });
    return {run:data.run_id,segment:data.segment_id,event:data.event_id,stamp,counts};
  }
  const api = {outside,advance};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SensorBands = api;
})(globalThis);
