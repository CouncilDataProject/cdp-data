-- How many minutes items and matters are in an event?
select e.id, b.name, e.event_datetime, count(mi.id) as minutes_items, count(m.id) as matters
from event e
join body b on b.id = e.body_id
left join event_minutes_item emi on emi.event_id = e.id
left join minutes_item mi on emi.minutes_item_id = mi.id
left join matter m on mi.matter_id = m.id
group by e.id
order by count(mi.id) desc

-- What are the common matter types?
select m.matter_type, count(*)
from matter m
group by m.matter_type
order by count(*) desc

-- What are the common matter statuses?
select ms.status, count(*)
from matter m
left join matter_status ms on ms.matter_id = m.id
group by ms.status
order by count(*) desc
