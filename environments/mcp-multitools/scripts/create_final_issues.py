#!/usr/bin/env python3
"""
Create Linear issues with minimal Supabase data + new meaningful content.
- 2-4 lines from tables (names, key IDs)
- Action steps, websites, extra context NOT in Supabase
- Only use employee ID when name is duplicate (e.g., James Chen)
- ~10% assigned to multiple people
"""

import os
import httpx
import asyncio
import random
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")
SUPABASE_HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

LINEAR_API_KEY = os.getenv("LINEAR_API_KEY")
LINEAR_API_URL = "https://api.linear.app/graphql"
LINEAR_HEADERS = {"Authorization": LINEAR_API_KEY, "Content-Type": "application/json"}

# Names that have duplicates in the database - need ID
DUPLICATE_NAMES = {"James Chen", "Alexander"}


async def query_supabase(table, select="*", limit=500):
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{SUPABASE_URL}/rest/v1/{table}?select={select}&limit={limit}", headers=SUPABASE_HEADERS)
        return resp.json()


async def delete_all_issues():
    query = "query { issues(first: 100) { nodes { id } } }"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(LINEAR_API_URL, headers=LINEAR_HEADERS, json={"query": query})
        issues = response.json().get("data", {}).get("issues", {}).get("nodes", [])
    
    for issue in issues:
        mutation = "mutation($id: String!) { issueDelete(id: $id) { success } }"
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(LINEAR_API_URL, headers=LINEAR_HEADERS, json={"query": mutation, "variables": {"id": issue["id"]}})
        await asyncio.sleep(0.03)
    print(f"Deleted {len(issues)} issues")


async def get_team_states():
    query = "query { teams { nodes { id states { nodes { id name } } } } }"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(LINEAR_API_URL, headers=LINEAR_HEADERS, json={"query": query})
        team = response.json().get("data", {}).get("teams", {}).get("nodes", [])[0]
    states = {s["name"]: s["id"] for s in team.get("states", {}).get("nodes", [])}
    return team["id"], states


async def create_issue(team_id, title, description, state_id=None):
    mutation = """
    mutation($title: String!, $description: String!, $teamId: String!, $stateId: String) {
        issueCreate(input: { title: $title, description: $description, teamId: $teamId, stateId: $stateId }) {
            success issue { identifier }
        }
    }
    """
    variables = {"title": title, "description": description, "teamId": team_id}
    if state_id:
        variables["stateId"] = state_id
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(LINEAR_API_URL, headers=LINEAR_HEADERS, json={"query": mutation, "variables": variables})
        return response.json().get("data", {}).get("issueCreate", {}).get("issue")


def emp_name(e, include_id=False):
    """Format employee name, only include ID if duplicate name."""
    name = f"{e['first_name']} {e['last_name']}"
    if include_id or any(dup in name for dup in DUPLICATE_NAMES):
        return f"{name} ({e['id']})"
    return name


def build_issues(employees, orders, customers, products, meetings, suppliers):
    emp_map = {e["id"]: e for e in employees}
    cust_map = {c["id"]: c for c in customers}
    
    pending = [o for o in orders if o["status"] == "pending"]
    processing = [o for o in orders if o["status"] == "processing"]
    cancelled = [o for o in orders if o["status"] == "cancelled"]
    delivered = [o for o in orders if o["status"] == "delivered"]
    shipped = [o for o in orders if o["status"] == "shipped"]
    
    out_of_stock = [p for p in products if p["stock_quantity"] == 0]
    low_stock = [p for p in products if 0 < p["stock_quantity"] < 20]
    
    sales = [e for e in employees if e["department_id"] == "DEPT-SAL" and e["is_active"]]
    eng = [e for e in employees if e["department_id"] == "DEPT-ENG" and e["is_active"]]
    ops = [e for e in employees if e["department_id"] == "DEPT-OPS" and e["is_active"]]
    support = [e for e in employees if e["department_id"] == "DEPT-SUP" and e["is_active"]]
    hr = [e for e in employees if e["department_id"] == "DEPT-HR" and e["is_active"]]
    fin = [e for e in employees if e["department_id"] == "DEPT-FIN" and e["is_active"]]
    
    issues = []
    
    # ===== DONE ISSUES =====
    
    if delivered:
        o = delivered[0]
        cust = cust_map.get(o["customer_id"], {})
        rep = emp_map.get(o["sales_rep_id"], {})
        issues.append({
            "title": f"Ship confirmation sent to {cust.get('company_name', 'customer')}",
            "description": f"""Sent tracking details for order {o['id']}.

Customer confirmed receipt via email. Left a positive review on Trustpilot - might be worth featuring on our testimonials page.

**Assigned to:** {emp_name(rep)}""",
            "status": "Done"
        })
    
    if suppliers:
        s = suppliers[0]
        issues.append({
            "title": f"Negotiated better terms with {s['company_name']}",
            "description": f"""Extended payment terms from Net 30 to Net 45. They agreed because of our order volume.

Should save us about $12K in cash flow improvements this quarter. Updated the supplier record in the system.

Might want to try the same approach with other suppliers rated above 4.0.

**Assigned to:** {ops[0]['first_name']} {ops[0]['last_name']}""" if ops else "Procurement",
            "status": "Done"
        })
    
    if sales:
        e = sales[0]
        issues.append({
            "title": f"Completed sales training for Q4",
            "description": f"""All sales reps completed the new CRM training module on learn.salesforce.com/advanced-pipeline.

{emp_name(e)} scored highest (94%) - maybe have them run a refresher for the team next month.

Training certificates uploaded to HR portal.

**Assigned to:** {emp_name(hr[0])}""" if hr else "HR Team",
            "status": "Done"
        })
    
    if customers:
        c = customers[0]
        issues.append({
            "title": f"Updated billing address for {c['company_name']}",
            "description": f"""They moved offices last month. New address is in the system.

Also updated their primary contact to {c.get('contact_first_name', 'the new')} {c.get('contact_last_name', 'contact')} since Janet left.

Let {emp_name(emp_map.get(c.get('account_manager_id'), sales[0] if sales else {}))} know so they can send a housewarming gift.

**Assigned to:** {emp_name(support[0])}""" if support else "Support",
            "status": "Done"
        })
    
    if eng:
        issues.append({
            "title": "Fixed the checkout timeout bug",
            "description": f"""The 504 errors on checkout were caused by the payment gateway taking too long. Added a retry mechanism with exponential backoff.

Deployed to prod yesterday, monitoring looks clean. Error rate down from 2.3% to 0.1%.

Docs updated at docs.internal/checkout-flow#retry-logic.

**Assigned to:** {emp_name(eng[0])}""",
            "status": "Done"
        })
    
    if delivered and len(delivered) > 1:
        o = delivered[1]
        cust = cust_map.get(o["customer_id"], {})
        issues.append({
            "title": f"{cust.get('company_name', 'Customer')} reorder processed",
            "description": f"""They liked the first order so much they placed another one right away (order {o['id']}).

Gave them a 5% loyalty discount - cleared it with management first. Good candidate for our annual contract program.

**Assigned to:** {emp_name(emp_map.get(o['sales_rep_id'], {}))}""",
            "status": "Done"
        })
    
    # ===== IN PROGRESS ISSUES =====
    
    for o in pending[:3]:
        cust = cust_map.get(o["customer_id"], {})
        rep = emp_map.get(o["sales_rep_id"], {})
        issues.append({
            "title": f"Waiting on payment: {cust.get('company_name', 'customer')}",
            "description": f"""Order {o['id']} stuck in pending since {o['order_date']}.

Stripe shows the charge as "pending verification" - might be a fraud flag. Called the customer and they said the payment should have gone through.

Next steps:
- Check with Stripe support (case #STR-2024-8834)
- If not resolved by EOD, offer bank transfer as alternative
- Keep customer updated

**Assigned to:** {emp_name(rep)}""",
            "status": "In Progress"
        })
    
    if out_of_stock:
        p = out_of_stock[0]
        sup = next((s for s in suppliers if s["id"] == p.get("supplier_id")), {})
        issues.append({
            "title": f"Urgent restock: {p['name']}",
            "description": f"""We've had 3 customers ask about {p['sku']} this week and it's showing out of stock.

Called {sup.get('company_name', 'the supplier')} - they can ship 50 units by Friday if we order today. Cost is $2 higher per unit due to rush shipping but worth it.

PO draft saved at drive.google.com/purchasing/po-drafts/2024-nov.

**Assigned to:** {emp_name(ops[0])}""" if ops else "Operations",
            "status": "In Progress"
        })
    
    if processing:
        o = processing[0]
        cust = cust_map.get(o["customer_id"], {})
        rep = emp_map.get(o["sales_rep_id"], {})
        # Multiple assignees example (~10%)
        second_person = eng[0] if eng else rep
        issues.append({
            "title": f"Custom packaging for {cust.get('company_name', 'customer')}",
            "description": f"""Order {o['id']} needs special handling - they requested eco-friendly packaging for a sustainability campaign.

We don't normally stock biodegradable packing peanuts but found a supplier on Alibaba. Lead time is 3 days.

Worth adding to our standard options? Getting quotes for bulk purchase.

**Assigned to:** {emp_name(rep)}, {emp_name(second_person)}""",
            "status": "In Progress"
        })
    
    if shipped:
        o = shipped[0]
        cust = cust_map.get(o["customer_id"], {})
        issues.append({
            "title": f"Tracking issue: {cust.get('company_name', 'customer')} can't see updates",
            "description": f"""Order {o['id']} shipped on {o.get('shipped_date', 'last week')} but tracking hasn't updated since leaving our warehouse.

FedEx says it's a scan issue - package is actually in transit. I've escalated to their support (ticket #FDX-445892).

Need to call customer and explain, maybe offer free shipping on next order as goodwill.

**Assigned to:** {emp_name(support[0])}""" if support else "Support Team",
            "status": "In Progress"
        })
    
    if sales and len(sales) > 1:
        issues.append({
            "title": "Territory conflict in APAC region",
            "description": f"""{emp_name(sales[0])} and {emp_name(sales[1])} are both working the same accounts in Singapore.

Had a call with both of them - agreeing to split by industry vertical instead of geography. Tech companies go to {sales[0]['first_name']}, manufacturing to {sales[1]['first_name']}.

Drawing up new territory map in Salesforce. Target completion: Friday.

**Assigned to:** {emp_name(sales[0])}, {emp_name(sales[1])}""",
            "status": "In Progress"
        })
    
    if meetings:
        m = meetings[0]
        org = emp_map.get(m["organizer_id"], {})
        issues.append({
            "title": f"Prep deck for {m['title']}",
            "description": f"""Meeting is on {m['scheduled_date']}. Need to prepare slides covering Q4 progress.

Pull numbers from the dashboard at analytics.internal/q4-summary. Include the chart showing month-over-month growth.

{emp_name(org)} will present but needs the deck by day before.

**Assigned to:** {emp_name(fin[0])}""" if fin else "Finance",
            "status": "In Progress"
        })
    
    # ===== TODO ISSUES =====
    
    for p in low_stock[:3]:
        sup = next((s for s in suppliers if s["id"] == p.get("supplier_id")), {})
        issues.append({
            "title": f"Reorder soon: {p['name']} ({p['stock_quantity']} left)",
            "description": f"""Getting low on {p['sku']}. Usually takes 2 weeks to restock from {sup.get('company_name', 'supplier')}.

Check last order quantity in PO system and adjust for holiday demand - probably need 20% more than usual.

**Assigned to:** {emp_name(ops[0])}""" if ops else "Operations",
            "status": "Todo"
        })
    
    for o in cancelled[:2]:
        cust = cust_map.get(o["customer_id"], {})
        rep = emp_map.get(o["sales_rep_id"], {})
        issues.append({
            "title": f"Win-back: {cust.get('company_name', 'Customer')} cancelled",
            "description": f"""Order {o['id']} was cancelled. Don't know why yet.

Worth a call to understand - if it's pricing, we have room to negotiate. If it's delivery time, we can offer expedited.

Check CRM notes first to see if there's history. Good accounts are worth fighting for.

**Assigned to:** {emp_name(rep)}""",
            "status": "Todo"
        })
    
    if customers and len(customers) > 5:
        c = customers[5]
        mgr = emp_map.get(c.get("account_manager_id"), {})
        issues.append({
            "title": f"Annual review: {c['company_name']}",
            "description": f"""Account is up for renewal next month. Currently on standard terms.

Pull their order history and see if volume justifies better pricing. Also check if they've had any support tickets - don't want surprises in the renewal call.

Template for renewal proposals is at templates.internal/renewals/enterprise.docx.

**Assigned to:** {emp_name(mgr)}""",
            "status": "Todo"
        })
    
    if suppliers and len(suppliers) > 1:
        s = suppliers[1]
        issues.append({
            "title": f"Backup supplier needed for {s.get('country', 'region')}",
            "description": f"""We're too dependent on {s['company_name']} for that market. If they have supply issues, we're stuck.

Find 2-3 alternatives on thomasnet.com or alibaba.com. Get quotes and samples before December.

Quality is key - don't want to compromise on that.

**Assigned to:** {emp_name(ops[0])}""" if ops else "Procurement",
            "status": "Todo"
        })
    
    if eng and support:
        issues.append({
            "title": "Customer portal login issues",
            "description": f"""Got 4 tickets this week about customers unable to reset passwords. The reset email goes to spam for Gmail users.

Need to:
- Check SPF/DKIM records at cloudflare.com dashboard
- Maybe switch to SendGrid for transactional emails
- Add "check spam folder" note to the login page for now

**Assigned to:** {emp_name(eng[0])}, {emp_name(support[0])}""",
            "status": "Todo"
        })
    
    if hr:
        issues.append({
            "title": "Update employee handbook for remote work",
            "description": f"""Current policy is outdated - written pre-COVID. Need to formalize the hybrid arrangement.

Legal reviewed and approved the draft. See comments in docs.google.com/handbook-v3-draft.

Target rollout: January 1st. Announce in December all-hands.

**Assigned to:** {emp_name(hr[0])}""",
            "status": "Todo"
        })
    
    if fin:
        issues.append({
            "title": "Q4 budget variance report",
            "description": f"""We're 8% over on marketing spend. Need to explain this to leadership.

The overage is mostly from the trade show in October - wasn't in original budget but generated good leads. Pull ROI data from HubSpot to justify.

Report template at finance.internal/templates/variance-q4.xlsx.

**Assigned to:** {emp_name(fin[0])}""",
            "status": "Todo"
        })
    
    if meetings and len(meetings) > 2:
        m = meetings[2]
        org = emp_map.get(m["organizer_id"], {})
        issues.append({
            "title": f"Book room for {m['title']}",
            "description": f"""Meeting on {m['scheduled_date']} needs a conference room. Preferably one with video setup since {org.get('first_name', 'the organizer')} mentioned some folks dialing in.

Check availability at rooms.internal/booking. Backup option: use Zoom if no rooms free.

**Assigned to:** Office Admin""",
            "status": "Todo"
        })
    
    if products:
        p = products[10] if len(products) > 10 else products[0]
        issues.append({
            "title": f"Photo update: {p['name']}",
            "description": f"""Product photos are outdated - using stock imagery. Need fresh shots for the website refresh.

Photography vendor (Aperture Studios, contact: mike@aperturestudios.com) quoted $200 per product. Maybe batch this with other items needing updates.

Current images at cdn.internal/products/{p['sku']}/

**Assigned to:** Marketing Team""",
            "status": "Todo"
        })
    
    if sales:
        issues.append({
            "title": "Sales playbook for enterprise deals",
            "description": f"""Need documented process for deals over $50K. Currently everyone does their own thing.

{emp_name(sales[0])} closed our biggest deal last quarter - have them write up their approach. Include objection handling, pricing flexibility, legal approval workflow.

Target: 10-page PDF at sales.internal/playbooks/enterprise.pdf

**Assigned to:** {emp_name(sales[0])}""",
            "status": "Todo"
        })
    
    if eng and len(eng) > 1:
        issues.append({
            "title": "Code review backlog",
            "description": f"""12 PRs waiting for review, some over a week old. Slowing down the team.

Proposal: rotate review duty weekly. This week: {emp_name(eng[0])}. Next week: {emp_name(eng[1])}.

Set up GitHub action to ping on Slack if PR is open >48 hours.

**Assigned to:** {emp_name(eng[0])}, {emp_name(eng[1])}""",
            "status": "Todo"
        })
    
    if customers and len(customers) > 10:
        c = customers[10]
        issues.append({
            "title": f"NPS survey follow-up: {c['company_name']}",
            "description": f"""They gave us a 6 (passive) on the last NPS survey. Left comment about slow response times.

Worth a personal call to understand better. If we can move them to promoter, they'd be good reference account.

Survey data at surveys.internal/nps/2024-q3/

**Assigned to:** {emp_name(emp_map.get(c.get('account_manager_id'), sales[0] if sales else {}))}""",
            "status": "Todo"
        })
    
    issues.append({
        "title": "Competitive analysis: new market entrant",
        "description": f"""Heard that NovaTech Solutions launched a competing product. Pricing seems aggressive.

Need someone to:
- Sign up for their trial
- Document features vs ours
- Check their reviews on G2 and Capterra

Summary deck due for strategy meeting on the 20th.

**Assigned to:** Product Team""",
        "status": "Todo"
    })
    
    issues.append({
        "title": "Broken link on pricing page",
        "description": f"""Customer reported that "Contact Sales" button leads to 404. Probably from the last deploy.

Quick fix - just need to update the href in the React component. File is probably src/pages/Pricing.tsx.

Low priority but looks unprofessional.

**Assigned to:** {emp_name(eng[0])}""" if eng else "Engineering",
        "status": "Todo"
    })
    
    if shipped and len(shipped) > 1:
        o = shipped[1]
        cust = cust_map.get(o["customer_id"], {})
        issues.append({
            "title": f"Delivery confirmation: {cust.get('company_name', 'Customer')}",
            "description": f"""Order {o['id']} shows delivered but customer hasn't confirmed receipt.

Might just be busy. Send a quick email checking if everything arrived OK. If no response in 3 days, call them.

Template at templates.internal/delivery-followup.html

**Assigned to:** {emp_name(emp_map.get(o['sales_rep_id'], {}))}""",
            "status": "Todo"
        })
    
    issues.append({
        "title": "Set up staging environment",
        "description": f"""We've been deploying straight to prod which is risky. Need a proper staging setup.

AWS estimate: ~$400/month for staging infra. Submitted budget request to finance.

Once approved, {eng[0]['first_name'] if eng else 'DevOps'} can provision in about 2 days.

**Assigned to:** {emp_name(eng[0])}""" if eng else "DevOps",
        "status": "Todo"
    })
    
    issues.append({
        "title": "Quarterly business review deck",
        "description": f"""QBR with exec team next Wednesday. Need slides covering:
- Revenue vs target
- Customer growth metrics
- Top deals closed
- Churn analysis

Deadline: Monday EOD so leadership can review.

**Assigned to:** {emp_name(fin[0])}, {emp_name(sales[0])}""" if fin and sales else "Finance & Sales",
        "status": "Todo"
    })
    
    return issues


async def main():
    print("=" * 60)
    print("Creating Final Connected Issues")
    print("=" * 60)
    
    print("\nQuerying Supabase...")
    employees = await query_supabase("employees")
    orders = await query_supabase("orders")
    customers = await query_supabase("customers")
    products = await query_supabase("products")
    meetings = await query_supabase("meetings")
    suppliers = await query_supabase("suppliers")
    
    print("\nDeleting existing issues...")
    await delete_all_issues()
    
    team_id, states = await get_team_states()
    status_map = {"Done": states.get("Done"), "In Progress": states.get("In Progress"), "Todo": states.get("Todo")}
    
    print("\nBuilding issues...")
    issues = build_issues(employees, orders, customers, products, meetings, suppliers)
    
    print(f"\nCreating {len(issues)} issues in Linear...")
    counts = {"Done": 0, "In Progress": 0, "Todo": 0}
    
    for issue in issues:
        result = await create_issue(team_id, issue["title"], issue["description"], status_map.get(issue["status"]))
        if result:
            emoji = {"Done": "✅", "In Progress": "🔄", "Todo": "📋"}[issue["status"]]
            print(f"  {emoji} {result['identifier']}: {issue['title'][:45]}...")
            counts[issue["status"]] += 1
        await asyncio.sleep(0.1)
    
    print(f"\n{'=' * 60}")
    print(f"Created {sum(counts.values())} issues")
    print(f"  ✅ Done: {counts['Done']}")
    print(f"  🔄 In Progress: {counts['In Progress']}")  
    print(f"  📋 Todo: {counts['Todo']}")


if __name__ == "__main__":
    asyncio.run(main())

