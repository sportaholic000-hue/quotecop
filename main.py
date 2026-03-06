"""
QuoteCop - AI Quote Checker SaaS
Production-ready FastAPI application that analyzes contractor/service quotes,
flags overcharges, and generates counter-negotiation scripts.
"""

import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import Optional

import stripe
import openai
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
APP_URL = os.getenv("APP_URL", "http://localhost:8000")
PRICE_ONE_TIME = os.getenv("STRIPE_PRICE_ONE_TIME", "")
PRICE_SUBSCRIPTION = os.getenv("STRIPE_PRICE_SUBSCRIPTION", "")

stripe.api_key = STRIPE_SECRET_KEY

# ---------------------------------------------------------------------------
# App Init
# ---------------------------------------------------------------------------
app = FastAPI(
    title="QuoteCop",
    description="AI-Powered Quote Checker - Stop Getting Overcharged",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# In-memory stores (swap for Redis / Postgres in production at scale)
# ---------------------------------------------------------------------------
demo_usage: dict[str, dict] = {}
paid_sessions: dict[str, dict] = {}
subscribers: set[str] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_fingerprint(request: Request) -> str:
    ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "unknown")
    raw = f"{ip}:{ua}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def check_demo_available(fingerprint: str) -> bool:
    usage = demo_usage.get(fingerprint)
    if usage is None:
        return True
    return usage.get("count", 0) < 1


def record_demo_usage(fingerprint: str):
    if fingerprint not in demo_usage:
        demo_usage[fingerprint] = {"count": 0, "first_used": datetime.utcnow().isoformat()}
    demo_usage[fingerprint]["count"] += 1


def check_paid_access(session_id: str) -> bool:
    session = paid_sessions.get(session_id)
    if not session:
        return False
    if session.get("type") == "subscription":
        return True
    if session.get("type") == "one_time":
        return session.get("uses_remaining", 0) > 0
    return False


def consume_paid_use(session_id: str):
    session = paid_sessions.get(session_id)
    if session and session.get("type") == "one_time":
        session["uses_remaining"] = max(0, session.get("uses_remaining", 0) - 1)


# ---------------------------------------------------------------------------
# OpenAI Analysis Engine
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are QuoteCop, an expert cost analyst and consumer advocate. You analyze service quotes and estimates that consumers receive from contractors, mechanics, dentists, plumbers, electricians, roofers, auto body shops, and other service providers.

Your job:
1. Parse every line item from the quote
2. Research fair market pricing for the given location and region
3. Flag any items that appear overpriced (more than 15 percent above market average)
4. Flag any suspicious or unnecessary items
5. Calculate total potential savings
6. Generate a polite but firm counter-negotiation script the user can copy-paste

IMPORTANT RULES:
- Be specific with dollar amounts and percentages
- Reference real market data and typical price ranges
- Be fair -- not every quote is a ripoff; acknowledge fair pricing too
- Consider regional cost-of-living differences
- Factor in complexity, urgency, and quality tiers
- If you cannot determine exact pricing, give reasonable ranges with LOW confidence

Return your analysis as valid JSON with this EXACT structure:
{
  "summary": {
    "verdict": "OVERPRICED" or "FAIR" or "GOOD_DEAL",
    "total_quoted": <number>,
    "estimated_fair_total": <number>,
    "potential_savings": <number>,
    "savings_percentage": <number>,
    "confidence": "HIGH" or "MEDIUM" or "LOW"
  },
  "line_items": [
    {
      "item": "<description>",
      "quoted_price": <number>,
      "fair_range_low": <number>,
      "fair_range_high": <number>,
      "status": "OVERPRICED" or "FAIR" or "GOOD_DEAL" or "SUSPICIOUS",
      "note": "<explanation>"
    }
  ],
  "red_flags": ["<list of concerns>"],
  "green_flags": ["<list of positive aspects>"],
  "negotiation_script": "<A complete, ready-to-use script the user can say or send to negotiate a better price>",
  "tips": ["<actionable tips for this type of service>"]
}"""


async def analyze_quote_with_ai(
    quote_text: str,
    service_type: str,
    location: str,
    user_api_key: Optional[str] = None,
) -> dict:
    api_key = user_api_key or OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured. Please provide your own key or contact support.")

    client = openai.AsyncOpenAI(api_key=api_key)

    user_message = f"""Analyze this {service_type} quote/estimate for a customer located in {location}:

--- QUOTE START ---
{quote_text}
--- QUOTE END ---

Provide your full analysis as JSON."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key. Please check your key and try again.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="API rate limit reached. Please wait a moment and try again.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------
class QuoteAnalysisRequest(BaseModel):
    quote_text: str
    service_type: str = "general"
    location: str = "United States"
    user_api_key: Optional[str] = None
    session_id: Optional[str] = None


class CheckoutRequest(BaseModel):
    plan: str  # "one_time" or "subscription"
    fingerprint: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes -- Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stripe_publishable_key": STRIPE_PUBLISHABLE_KEY,
        "app_url": APP_URL,
    })


@app.get("/success", response_class=HTMLResponse)
async def success_page(request: Request, session_id: str = ""):
    return templates.TemplateResponse("success.html", {
        "request": request,
        "session_id": session_id,
        "stripe_publishable_key": STRIPE_PUBLISHABLE_KEY,
        "app_url": APP_URL,
    })


# ---------------------------------------------------------------------------
# Routes -- API
# ---------------------------------------------------------------------------
@app.post("/api/analyze-quote")
async def analyze_quote(request: Request, body: QuoteAnalysisRequest):
    if not body.quote_text or len(body.quote_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Please provide a more detailed quote (at least 20 characters).")
    if len(body.quote_text) > 10000:
        raise HTTPException(status_code=400, detail="Quote text is too long. Maximum 10,000 characters.")

    fingerprint = get_fingerprint(request)
    has_paid = False

    if body.session_id and check_paid_access(body.session_id):
        has_paid = True
    elif body.user_api_key:
        has_paid = True  # BYOK -- user pays their own OpenAI costs
    elif not check_demo_available(fingerprint):
        return JSONResponse(status_code=402, content={
            "error": "demo_exhausted",
            "message": "Your free analysis has been used. Unlock unlimited checks for $9.99/mo or pay $2.99 for one more.",
        })

    result = await analyze_quote_with_ai(
        quote_text=body.quote_text,
        service_type=body.service_type,
        location=body.location,
        user_api_key=body.user_api_key,
    )

    if has_paid and body.session_id:
        consume_paid_use(body.session_id)
    elif not body.user_api_key:
        record_demo_usage(fingerprint)

    return JSONResponse(content={
        "success": True,
        "analysis": result,
        "demo_remaining": 0 if not has_paid and not body.user_api_key else None,
        "timestamp": datetime.utcnow().isoformat(),
    })


@app.get("/api/demo-status")
async def demo_status(request: Request):
    fingerprint = get_fingerprint(request)
    available = check_demo_available(fingerprint)
    return {"demo_available": available, "fingerprint": fingerprint[:8]}


# ---------------------------------------------------------------------------
# Routes -- Stripe Payments
# ---------------------------------------------------------------------------
@app.post("/create-checkout-session")
async def create_checkout_session(request: Request, body: CheckoutRequest):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe is not configured. Please contact support.")

    if body.plan == "one_time":
        price_id = PRICE_ONE_TIME
        mode = "payment"
    elif body.plan == "subscription":
        price_id = PRICE_SUBSCRIPTION
        mode = "subscription"
    else:
        raise HTTPException(status_code=400, detail="Invalid plan type.")

    if not price_id:
        try:
            if body.plan == "one_time":
                price = stripe.Price.create(
                    unit_amount=299,
                    currency="usd",
                    product_data={"name": "QuoteCop - Single Quote Check"},
                )
            else:
                price = stripe.Price.create(
                    unit_amount=999,
                    currency="usd",
                    recurring={"interval": "month"},
                    product_data={"name": "QuoteCop Pro - Unlimited Monthly"},
                )
            price_id = price.id
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Payment setup failed: {str(e)}")

    session_token = str(uuid.uuid4())

    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode=mode,
            success_url=f"{APP_URL}/success?session_id={session_token}",
            cancel_url=f"{APP_URL}/#pricing",
            metadata={"session_token": session_token, "plan": body.plan},
            allow_promotion_codes=True,
        )
        return {"checkout_url": checkout_session.url, "session_token": session_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Checkout failed: {str(e)}")


@app.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        except stripe.error.SignatureVerificationError:
            raise HTTPException(status_code=400, detail="Invalid webhook signature")
    else:
        event = json.loads(payload)

    event_type = event.get("type", "")

    if event_type == "checkout.session.completed":
        session_data = event["data"]["object"]
        metadata = session_data.get("metadata", {})
        session_token = metadata.get("session_token", "")
        plan = metadata.get("plan", "one_time")

        if session_token:
            if plan == "subscription":
                paid_sessions[session_token] = {
                    "type": "subscription",
                    "created": datetime.utcnow().isoformat(),
                    "customer_email": session_data.get("customer_email", ""),
                    "stripe_customer": session_data.get("customer", ""),
                }
                if session_data.get("customer_email"):
                    subscribers.add(session_data["customer_email"])
            else:
                paid_sessions[session_token] = {
                    "type": "one_time",
                    "uses_remaining": 1,
                    "created": datetime.utcnow().isoformat(),
                    "customer_email": session_data.get("customer_email", ""),
                }

    elif event_type == "customer.subscription.deleted":
        sub = event["data"]["object"]
        customer_email = sub.get("customer_email", "")
        subscribers.discard(customer_email)
        for token, data in paid_sessions.items():
            if data.get("customer_email") == customer_email and data.get("type") == "subscription":
                data["type"] = "cancelled"

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Health / Meta
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "QuoteCop",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/stats")
async def stats():
    return {
        "total_demo_users": len(demo_usage),
        "total_paid_sessions": len(paid_sessions),
        "total_subscribers": len(subscribers),
    }


# ---------------------------------------------------------------------------
# Development runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)